#pragma once

#include <sycl/sycl.hpp>

#define _BITS 4
#define _RADIX (1 << _BITS)
#define OFFSET 0

template <class T>
void radix_sort(sycl::queue q, sycl::buffer<T> buf) {
    const int work_groups = 64;
    const int n = buf.size();
    sycl::buffer<unsigned int> buf2{n};
    sycl::buffer<unsigned int> hist{n * _RADIX};
    sycl::buffer<unsigned int> gsum{(n + work_groups - 1) / work_groups};
    for (int pass = 0; pass < sizeof(T) * 8 / _BITS; pass++) {
        q.submit([&] (sycl::handler &cgh) { // histogram
            sycl::local_accessor<unsigned int> loc_histo{_RADIX * work_groups, cgh};
            sycl::accessor d_Histograms{hist, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor d_Keys{buf, cgh, sycl::read_only};
            cgh.parallel_for(sycl::nd_range<1>{n, work_groups}, [=] (sycl::nd_item<1> item) {
                int it = item.get_local_id(0);  // i local number of the processor
                int ig = item.get_global_id(0); // global number = i + g I
                int gr = item.get_group(0); // gr group number
                int groups = item.get_group_range(0);
                int items  = item.get_local_range(0);

                // initialize the local histograms to zero
                for(int ir = 0; ir < _RADIX; ir++) {
                    loc_histo[ir * items + it] = 0;
                }

                item.barrier(sycl::access::fence_space::local_space);

                // range of keys that are analyzed by the work item
                int sublist_size  = n/groups/items; // size of the sub-list
                int sublist_start = ig * sublist_size; // beginning of the sub-list

                // compute the index
                // the computation depends on the transposition
                for(int j = 0; j < sublist_size; j++) {
                    int k = j + sublist_start;

                    T key = d_Keys[k] + OFFSET;

                    // extract the group of _BITS bits of the pass
                    // the result is in the range 0.._RADIX-1
                    // _BITS = size of _RADIX in bits. So basically they
                    // represent both the same. 
                    T shortkey=(( key >> (pass * _BITS)) & (_RADIX-1)); // _RADIX-1 to get #_BITS "ones"

                    // increment the local histogram
                    loc_histo[shortkey *  items + it ]++;
                }

                // wait for local histogram to finish
                item.barrier(sycl::access::fence_space::local_space);

                // copy the local histogram to the global one
                // in this case the global histo is the group histo.
                for (int ir = 0; ir < _RADIX; ir++) {
                    d_Histograms[items * (ir * groups + gr) + it] = loc_histo[ir * items + it];
                }
            });
        });
        q.submit([&] (sycl::handler &cgh) { // scanhistogram
            sycl::local_accessor<unsigned int> temp{_RADIX * work_groups, cgh};
            sycl::accessor histo{hist, cgh, sycl::read_write};
            sycl::accessor globsum{gsum, cgh, sycl::write_only, sycl::no_init};
            cgh.parallel_for(sycl::nd_range<1>{n, work_groups}, [=] (sycl::nd_item<1> item) {
                int it = item.get_local_id(0);
                int ig = item.get_global_id(0);
                int decale = 1;
                int n = item.get_local_range(0) << 1;
                int gr = item.get_group(0);

                // load input into local memory
                // up sweep phase
                temp[(it << 1)]     = histo[(ig << 1)];
                temp[(it << 1) + 1] = histo[(ig << 1) + 1];

                // parallel prefix sum (algorithm of Blelloch 1990)
                // This loop runs log2(n) times
                for (int d = n >> 1; d > 0; d >>= 1) {
                    item.barrier(sycl::access::fence_space::local_space);
                    if (it < d) {
                        int ai = decale * ((it << 1) + 1) - 1;
                        int bi = decale * ((it << 1) + 2) - 1;
                        temp[bi] += temp[ai];
                    }
                    decale <<= 1;
                }

                // store the last element in the global sum vector
                // (maybe used in the next step for constructing the global scan)
                // clear the last element
                if (it == 0) {
                    globsum[gr] = temp[n - 1];
                    temp[n - 1] = 0;
                }

                // down sweep phase
                // This loop runs log2(n) times
                for (int d = 1; d < n; d <<= 1){
                    decale >>= 1;
                    item.barrier(sycl::access::fence_space::local_space);

                    if (it < d){
                        int ai = decale*((it << 1) + 1) - 1;
                        int bi = decale*((it << 1) + 2) - 1;

                        int t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                // write results to device memory
                histo[(ig << 1)]       = temp[(it << 1)];
                histo[(ig << 1) + 1]   = temp[(it << 1) + 1];
            });
        });
        q.submit([&] (sycl::handler &cgh) { // reorder
            sycl::local_accessor<T> loc_histo{_RADIX * work_groups, cgh};
            sycl::accessor d_Histograms{hist, cgh, sycl::read_write};
            sycl::accessor d_inKeys{buf, cgh, sycl::read_only};
            sycl::accessor d_outKeys{buf2, cgh, sycl::write_only, sycl::no_init};
            cgh.parallel_for(sycl::nd_range<1>{n, work_groups}, [=] (sycl::nd_item<1> item) {
                int it = item.get_local_id(0);  // i local number of the processor
                int ig = item.get_global_id(0); // global number = i + g I
                int gr = item.get_group(0);				// gr group number
                int groups = item.get_group_range(0);	// G: group count
                int items = item.get_local_range(0);			// group size

                int start = ig *(n / groups / items);   // index of first elem this work-item processes
                int size  = n / groups / items;			// count of elements this work-item processes

                // take the histogram in the cache
                for (int ir = 0; ir < _RADIX; ir++){
                    loc_histo[ir * items + it] = d_Histograms[items * (ir * groups + gr) + it];
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int j = 0; j < size; j++) {
                    int k = j + start;
                    T key = d_inKeys[k] + OFFSET;
                    T shortkey = ((key >> (pass * _BITS)) & (_RADIX - 1));	// shift element to relevant bit positions

                    int newpos = loc_histo[shortkey * items + it];

                    d_outKeys[newpos] = key - OFFSET;

                    newpos++;
                    loc_histo[shortkey * items + it] = newpos;
                }
            });
        });
    }
}
