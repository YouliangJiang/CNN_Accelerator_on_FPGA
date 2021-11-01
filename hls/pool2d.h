#pragma once
#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
#include "sliding_window_unit.h"
#include "stream_tools.h"



/**
 * pool层计算处理函数
 */
template <	unsigned K,
			unsigned IN_CH,
			unsigned IN_BIT,
            unsigned VEC_NUMS>
void pool_cal(
	stream<ap_uint<IN_CH*IN_BIT> >& vec,
	stream<ap_uint<IN_CH*IN_BIT> >& out,
	const unsigned reps)
{
	ap_uint<IN_CH*IN_BIT> result = 0;
	unsigned k_cnt = 0;

	for (unsigned rep = 0; rep < reps*VEC_NUMS; rep++) {
#pragma HLS PIPELINE II=1

        // 这里的temp_vec应该是寄存器（reg）类型
		ap_uint<IN_CH*IN_BIT> temp_vec = vec.read();

		for (unsigned c = 0; c < IN_CH; c++) {
#pragma HLS UNROLL
			ap_uint<IN_BIT> temp = temp_vec( (c+1)*IN_BIT-1 , c*IN_BIT );
			result( (c+1)*IN_BIT-1, c*IN_BIT ) = (temp > result( (c+1)*IN_BIT-1, c*IN_BIT )) ? temp : result( (c+1)*IN_BIT-1, c*IN_BIT );
		}

        if(++ k_cnt == K*K) {
            out.write(result);
            result = 0;
            k_cnt = 0;
        }
	}
}

/**
 * 池化层
 */
template <	unsigned K,                 // kernel
			// unsigned S,                 // stride
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT>
void max_pool2d(
	stream<ap_uint<IN_CH*IN_BIT> >& in,
	stream<ap_uint<IN_CH*IN_BIT> >& out, 
	const unsigned reps)
{
#pragma HLS DATAFLOW

    const unsigned OUT_ROW = IN_ROW / 2;
    const unsigned OUT_COL = IN_COL / 2; 
    const unsigned S = 2;

    // 产生滑动窗口数据
    hls::stream<ap_uint<IN_CH*IN_BIT> > swu_out("swu_out");
    SWU<K, S, IN_ROW, IN_COL, IN_CH, IN_BIT>(in, swu_out, reps);

    // 处理数据
	// POOL<IN_ROW*IN_COL, Ibit, K, Cin, 1>(swu_out, out, reps);
    pool_cal<K, IN_CH, IN_BIT, OUT_ROW*OUT_COL*K*K>(swu_out, out, reps);
}


/* kernel_size=3, stride=2 */
template<	unsigned K, 
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT>
void maxPooling3x3(		hls::stream<ap_uint<IN_CH*IN_BIT> >& in,
						hls::stream<ap_uint<IN_CH*IN_BIT> >& out,
						const unsigned reps){

#pragma HLS DATAFLOW

	const unsigned OUT_ROW = (IN_ROW - 3)/2 + 1;
    const unsigned OUT_COL = (IN_COL - 3)/2 + 1;
    const unsigned S = 2;

    // 产生滑动窗口数据
    hls::stream<ap_uint<IN_CH*IN_BIT> > swu_out("swu_out");
    SWU<K, S, IN_ROW, IN_COL, IN_CH, IN_BIT>(in, swu_out, reps);

    // 处理数据
	// POOL<IN_ROW*IN_COL, Ibit, K, Cin, 1>(swu_out, out, reps);
    pool_cal<K, IN_CH, IN_BIT, OUT_ROW*OUT_COL*K*K>(swu_out, out, reps);
}


/*********************** fixed pooling ************************/

template <	unsigned K,
			unsigned IN_CH,
			unsigned IN_BIT,
            unsigned VEC_NUMS,
			unsigned SIMD>
void pool_cal_c24(
	stream<ap_uint<SIMD*IN_BIT> >& vec,
	stream<ap_uint<SIMD*IN_BIT> >& out,
	const unsigned reps)
{
	ap_uint<IN_CH*IN_BIT> temp_vec;
	ap_uint<IN_CH*IN_BIT> result = 0;
	unsigned k_cnt = 0;

	for (unsigned rep = 0; rep < reps*VEC_NUMS; rep++) {
#pragma HLS PIPELINE II=1

        // 这里的temp_vec应该是寄存器（reg）类型
		// ap_uint<IN_CH*IN_BIT> temp_vec = vec.read();
		for(unsigned i=0; i<(IN_CH/SIMD); i++){
			temp_vec((i+1)*IN_BIT*SIMD-1, i*IN_BIT*SIMD) = vec.read();
		}

		for (unsigned c = 0; c < IN_CH; c++) {
#pragma HLS UNROLL
			ap_uint<IN_BIT> temp = temp_vec( (c+1)*IN_BIT-1 , c*IN_BIT );
			result( (c+1)*IN_BIT-1, c*IN_BIT ) = (temp > result( (c+1)*IN_BIT-1, c*IN_BIT )) ? temp : result( (c+1)*IN_BIT-1, c*IN_BIT );
		}

        if(++ k_cnt == K*K) {
			for(unsigned j=0; j<(IN_CH/SIMD); j++){
				out.write(result((j+1)*IN_BIT*SIMD-1, j*IN_BIT*SIMD));
			}
            // out.write(result);
            result = 0;
            k_cnt = 0;
        }
	}
}

template <	unsigned K,                 // kernel
			// unsigned S,                 // stride
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,
			unsigned SIMD>
void max_pool2d_c24(
	stream<ap_uint<SIMD*IN_BIT> >& in,
	stream<ap_uint<SIMD*IN_BIT> >& out, 
	const unsigned reps)
{
#pragma HLS DATAFLOW

    const unsigned OUT_ROW = IN_ROW / 2;
    const unsigned OUT_COL = IN_COL / 2; 
    const unsigned S = 2;

    // 产生滑动窗口数据
    hls::stream<ap_uint<SIMD*IN_BIT> > swu_out("swu_out");
    SWU_c24_pool<K, S, IN_ROW, IN_COL, IN_CH, IN_BIT, SIMD>(in, swu_out, reps);

    // 处理数据
	// POOL<IN_ROW*IN_COL, Ibit, K, Cin, 1>(swu_out, out, reps);
    pool_cal_c24<K, IN_CH, IN_BIT, OUT_ROW*OUT_COL*K*K, SIMD>(swu_out, out, reps);
}