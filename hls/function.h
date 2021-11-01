#pragma once
#include "stream_tools.h"

/**
 *  padding 函数
 */ 
template <	unsigned IN_ROW,
			unsigned IN_COL,
            unsigned IN_CH,
			unsigned IN_BIT, 
			unsigned P>
void padding(
    // 将每一数竖看成一个元素
	stream<ap_uint<IN_CH*IN_BIT> >& in,
	stream<ap_uint<IN_CH*IN_BIT> >& out,
	const unsigned reps)
{
    const unsigned OUT_ROW = IN_ROW + 2 * P;
    const unsigned OUT_COL = IN_COL + 2 * P;

	ap_uint<IN_CH*IN_BIT> temp_out = 0;

	for (unsigned rep = 0; rep < reps; rep++) {

		for (unsigned h = 0; h < P; h++) {
			for (unsigned s = 0; s < OUT_COL; s++) {
				out.write(0);
			}
		}

		for (unsigned h = 0; h < IN_ROW; h++) {

			for ( unsigned s = 0; s < OUT_COL; s++ ) {
#pragma HLS PIPELINE II=1

				if ( (s < P) || (s >= OUT_COL-P) ) {
					temp_out = 0;
				}
				else {
					temp_out = in.read();
				}
				out.write(temp_out);
			}
		}

		for (unsigned h = 0; h < P; h++) {
			for (unsigned i = 0; i < OUT_COL; i++) {
				out.write(0);
			}
		}

	}
}

/************************ fixed padding ***************************/
template <	unsigned IN_ROW,
			unsigned IN_COL,
            unsigned IN_CH,
			unsigned IN_BIT, 
			unsigned P,
			unsigned SIMD>
void padding_c31(
    // 将每一数竖看成一个元素
	stream<ap_uint<SIMD*IN_BIT> >& in,
	stream<ap_uint<SIMD*IN_BIT> >& out,
	const unsigned reps )
{
    const unsigned OUT_ROW = IN_ROW + 2 * P;
    const unsigned OUT_COL = IN_COL + 2 * P;

	ap_uint<IN_CH*IN_BIT> temp_in = 0;
	ap_uint<IN_CH*IN_BIT> temp_out = 0;

	for (unsigned rep = 0; rep < reps; rep++) {

		for (unsigned h = 0; h < P; h++) {
			for (unsigned s = 0; s < OUT_COL; s++) {
				for(unsigned j=0; j<(IN_CH/SIMD); j++){
					out.write(0);
				}
				
			}
		}

		for (unsigned h = 0; h < IN_ROW; h++) {

			for ( unsigned s = 0; s < OUT_COL; s++ ) {
			#pragma HLS PIPELINE II=1

				if ( (s < P) || (s >= OUT_COL-P) ) {
					temp_out = 0;
				}
				else {
					for(unsigned tile_in=0; tile_in<(IN_CH/SIMD); tile_in++){
						temp_in((tile_in+1)*SIMD*IN_BIT-1, tile_in*IN_BIT*SIMD) = in.read();
					}
					temp_out = temp_in;
				}

				for(unsigned tile_out=0; tile_out<(IN_CH/SIMD); tile_out++){
					out.write(temp_out((tile_out+1)*SIMD*IN_BIT-1, tile_out*IN_BIT*SIMD));
				}
				// out.write(temp_out);
			}

			// for(unsigned s=0, o=0, cnt=0; o<(OUT_COL*IN_CH/SIMD); o++){
			// #pragma HLS PIPELINE II=1

			// 	if((s < P) || (s >= OUT_COL-P)){
			// 		temp_out = 0;
			// 	}
			// 	else{
			// 		temp_out = in.read();
			// 	}

			// 	if(++cnt == IN_CH/SIMD){
			// 		cnt = 0;
			// 		s++;
			// 	}

			// 	out.write(temp_out);
			// }
		}

		for (unsigned h = 0; h < P; h++) {
			for (unsigned i = 0; i < OUT_COL; i++) {
				for(unsigned j=0; j<(IN_CH/SIMD); j++){
					out.write(0);
				}
			}
		}

	}
}

template <	unsigned IN_ROW,
			unsigned IN_COL,
            unsigned IN_CH,
			unsigned IN_BIT, 
			unsigned P,
			unsigned SIMD,
			unsigned PE_LAST>
void padding_c42(
    // 将每一数竖看成一个元素
	stream<ap_uint<PE_LAST*IN_BIT> >& in,
	stream<ap_uint<SIMD*IN_BIT> >& out,
	const unsigned reps )
{
    const unsigned OUT_ROW = IN_ROW + 2 * P;
    const unsigned OUT_COL = IN_COL + 2 * P;

	ap_uint<IN_CH*IN_BIT> temp_in = 0;
	ap_uint<IN_CH*IN_BIT> temp_out = 0;

	for (unsigned rep = 0; rep < reps; rep++) {

		for (unsigned h = 0; h < P; h++) {
			for (unsigned s = 0; s < OUT_COL; s++) {
				for(unsigned j=0; j<(IN_CH/SIMD); j++){
					out.write(0);
				}
				
			}
		}

		for (unsigned h = 0; h < IN_ROW; h++) {

			for ( unsigned s = 0; s < OUT_COL; s++ ) {
			#pragma HLS PIPELINE II=1

				if ( (s < P) || (s >= OUT_COL-P) ) {
					temp_out = 0;
				}
				else {
					for(unsigned tile_in=0; tile_in<(IN_CH/PE_LAST); tile_in++){
						temp_in((tile_in+1)*PE_LAST*IN_BIT-1, tile_in*IN_BIT*PE_LAST) = in.read();
					}
					temp_out = temp_in;
				}

				for(unsigned tile_out=0; tile_out<(IN_CH/SIMD); tile_out++){
					out.write(temp_out((tile_out+1)*SIMD*IN_BIT-1, tile_out*IN_BIT*SIMD));
				}
				// out.write(temp_out);
			}
		}

		for (unsigned h = 0; h < P; h++) {
			for (unsigned i = 0; i < OUT_COL; i++) {
				for(unsigned j=0; j<(IN_CH/SIMD); j++){
					out.write(0);
				}
			}
		}

	}
}
