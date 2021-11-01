#include "stream_tools.h"
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_video.h>
#include "pool2d.h"
#include "function.h"
#include "atss_0426_round.h"
#include "math.h"


// #define DEBUG

using namespace hls;
using namespace std;

/****************************** Resize *******************************/

#define IN_IMAGE_WIDTH  640
#define IN_IMAGE_HEIGHT 360

#define RESIZE_IMAGE_WIDTH 352
#define RESIZE_IMAGE_HEIGHT 192

void stream_to_mat (hls::stream<ap_uint<24> >&in,
		 hls::Mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, HLS_8UC3> & raw_img) {
    
	for (int i=0; i<IN_IMAGE_HEIGHT; i++) {
		for (int j=0; j<IN_IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
            hls::Scalar<3, ap_uint<8> > pix;
            ap_uint<24> in_data = in.read();
            for (unsigned int p=0; p < 3; p ++) {
                
                pix.val[p] = in_data(8*p+7, 8*p);
            }
			raw_img << pix;
		}	
	}

}

void mat_to_stream (hls::Mat<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, HLS_8UC3> & resize_img,
                    hls::stream<ap_uint<24> >& out ) {
    
	for (int i=0; i<RESIZE_IMAGE_HEIGHT; i++) {
		for (int j=0; j<RESIZE_IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
            hls::Scalar<3, ap_uint<8> > pix;
            resize_img >> pix;
            ap_uint<24> out_data;
            for (unsigned int p=0; p < 3; p ++) {
                out_data(8*p+7, 8*p) = pix.val[p];
            }
            out.write(out_data);
		}	
	}

}

void resize(hls::stream<ap_uint<24> >&in, hls::stream<ap_uint<24> >& out) {
#pragma HLS dataflow
    hls::Mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, HLS_8UC3> raw_img;
#pragma HLS STREAM variable=raw_img depth=64 dim=1
    hls::Mat<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, HLS_8UC3> resize_img;
#pragma HLS STREAM variable=resize_img depth=64 dim=1
    stream_to_mat(in, raw_img);
    hls::Resize(raw_img, resize_img, HLS_INTER_LINEAR);
    // hls::Resize_opr_linear(raw_img, resize_img);
    mat_to_stream(resize_img, out);
}

void resize_batch(hls::stream<ap_uint<24> >& in, hls::stream<ap_uint<24> >& out, unsigned int reps) {
    for (unsigned int rep=0; rep < reps; rep ++) {
        resize(in, out);
    }
}

/****************************** Resize *******************************/

/************************ Image Normalization ************************/

const ap_fixed<16, 3, AP_RND> img_norm_weight[3] = {
    (ap_fixed<16, 3, AP_RND>)(1.0/58.395),
    (ap_fixed<16, 3, AP_RND>)(1.0/57.12),
    (ap_fixed<16, 3, AP_RND>)(1.0/57.375)
};
const ap_fixed<16, 3, AP_RND> img_norm_bias[3] = {
    (ap_fixed<16, 3, AP_RND>)(-123.675/58.395),
    (ap_fixed<16, 3, AP_RND>)(-116.28/57.12),
    (ap_fixed<16, 3, AP_RND>)(-103.53/57.375)
};

template <int IL_IN, int FL_IN, int IL_OUT, int FL_OUT>
ap_uint<(IL_OUT+FL_OUT)> truncate_img_norm(
    ap_uint<(IL_IN+FL_IN)> in) {
    
    // ap_fixed<(IL_IN+FL_IN), IL_IN, AP_RND> in_fixed = *(ap_fixed<(IL_IN+FL_IN), IL_IN, AP_RND>*)&in;
    // ap_uint<(IL_OUT+FL_OUT)> out;

    // ap_fixed<(IL_OUT+FL_OUT), IL_OUT, AP_RND> out_fixed;

    // if(in_fixed > (ap_fixed<(IL_OUT+FL_OUT), IL_OUT, AP_RND>)(pow(2, IL_OUT-1) - 1.0/pow(2, FL_OUT))){
    //     out_fixed = (ap_fixed<(IL_OUT+FL_OUT), IL_OUT, AP_RND>)(pow(2, IL_OUT-1) - 1.0/pow(2, FL_OUT));
    // }
    // else if(in_fixed < (ap_fixed<(IL_OUT+FL_OUT), IL_OUT, AP_RND>)(-pow(2, IL_OUT-1))){
    //     out_fixed = (ap_fixed<(IL_OUT+FL_OUT), IL_OUT, AP_RND>)(-pow(2, IL_OUT-1));
    // }
    // else{
    //     out_fixed = (ap_fixed<(IL_OUT+FL_OUT), IL_OUT, AP_RND>)in_fixed;
    // }

    // out = *(ap_uint<(IL_OUT+FL_OUT)>*)&out_fixed;
    // return out;

    ap_int<(IL_OUT+FL_OUT)> out;
    ap_int<(IL_OUT+FL_OUT+1)> out_tmp;

    out_tmp = in >> (IL_IN+FL_IN-IL_OUT-FL_OUT-1);

    if(out_tmp > 0){
        if(out_tmp < ((1<<(IL_OUT+FL_OUT))-1)){
            out_tmp += 1;
        }
    }
    // else{
    //     if(out != -(1<<(IL_OUT+FL_OUT))){
    //         out -= 1;
    //     }
    // }

    out = out_tmp >> 1;

    return out;
}

template <int IL_IN, int FL_IN, int IL_OUT, int FL_OUT>
ap_uint<(IL_OUT+FL_OUT)> truncate_unsigned(
    ap_uint<(IL_IN+FL_IN)> in) {

    ap_uint<(IL_OUT+FL_OUT)> out;
    ap_uint<(IL_IN+FL_OUT+1)> out_tmp;

    out_tmp = in >> (FL_IN-FL_OUT-1);

    if(out_tmp < ((1<<(IL_OUT+FL_OUT+1)) - 1)){
        out_tmp += 1;
    }
    else{
        out_tmp = (1<<(IL_OUT+FL_OUT+1)) - 1;
    }

    out = out_tmp >> 1;

    return out;
}

// template <int IL_IN, int FL_IN, int IL_OUT, int FL_OUT>
// ap_uint<(IL_OUT+FL_OUT)> truncate_unsigned_fl0(
//     ap_uint<(IL_IN+FL_IN)> in) {

//     ap_fixed<(IL_IN+FL_IN), IL_IN, AP_RND> in_fixed = *(ap_fixed<(IL_IN+FL_IN), IL_IN, AP_RND>*)&in;
//     ap_uint<IL_IN> in_int = in(IL_IN+FL_IN-1, FL_IN);
//     ap_uint<FL_IN> in_decimal = in(FL_IN-1, 0);

//     ap_int<IL_OUT> out_int;
//     ap_uint<(IL_OUT+FL_OUT)> out;


//         if(in_fixed >= ((1<<(IL_OUT-1)) -1)){
//             out_int = (1<<(IL_OUT-1)) -1;
//         }
//         else{
//             out_int = in_int;
//         }
//         out((IL_OUT+FL_OUT-1), FL_OUT) = out_int;


//     return out;
// }

template <int BIT_IN, int SIMD>
ap_uint<BIT_IN*SIMD> img_norm_calc(
    ap_uint<BIT_IN*SIMD> in,
    const ap_fixed<16, 3, AP_RND> weights[3],
    const ap_fixed<16, 3, AP_RND> bias[3]){

    ap_uint<BIT_IN*SIMD> res_out = 0;

    for(int i=0; i<SIMD; i++){
    #pragma HLS UNROLL
        ap_uint<BIT_IN> temp_in = in(BIT_IN*(i+1)-1, BIT_IN*i);
        ap_fixed<16, 5, AP_RND> temp_res = temp_in*weights[i] + bias[i];
        ap_uint<16> temp_res_uint = *(ap_uint<16>*)&temp_res;
        ap_uint<8> res_truncated = truncate_img_norm<5, 11, 5, 3>(temp_res_uint);
        res_out(BIT_IN*(i+1)-1, BIT_IN*i) = res_truncated;
    }

    return res_out;
}

template <int BIT_IN, int SIMD, int IMG_ROW, int IMG_COL>
void img_norm(  hls::stream<ap_uint<BIT_IN*SIMD> >& in,
                hls::stream<ap_uint<BIT_IN*SIMD> >& out,
                const unsigned int reps
 ){

// #pragma HLS DATAFLOW

    const unsigned loop_num = IMG_ROW*IMG_COL*reps;

    for(int i=0; i<loop_num; i++){
        #pragma HLS PIPELINE II=1
        ap_uint<BIT_IN*SIMD> in_read = in.read();
        ap_uint<BIT_IN*SIMD> out_buf = img_norm_calc<BIT_IN, SIMD>(in_read, img_norm_weight, img_norm_bias);
        out.write(out_buf);
    }
}

/************************ Image Normalization ************************/

template <  int BIT_IN,
            int BIT_W,
            int BIT_ACC,
            int SIMD>
ap_int<BIT_ACC> chi_vector_dot_product(
    ap_uint<BIT_IN*SIMD> in,
    ap_uint<BIT_W*SIMD> weight){

    ap_int<BIT_ACC> accumulation = 0;

    for(int i=0; i<SIMD; i++){
        #pragma HLS UNROLL
        ap_int<BIT_W> temp_w = weight((i+1)*BIT_W-1, i*BIT_W);
        ap_int<BIT_IN> temp_in = in((i+1)*BIT_IN-1, i*BIT_IN);
        ap_int<(BIT_W+BIT_IN)> result = temp_in * temp_w;
        accumulation += result;
    }
    return accumulation;
}

template <  int BIT_IN,
            int BIT_W,
            int BIT_ACC,
            int SIMD>
ap_int<BIT_ACC> chi_vector_dot_product_unsigned(
    ap_uint<BIT_IN*SIMD> in,
    ap_uint<BIT_W*SIMD> weight){

    ap_int<BIT_ACC> accumulation = 0;

    for(int i=0; i<SIMD; i++){
        #pragma HLS UNROLL
        ap_int<BIT_W> temp_w = weight((i+1)*BIT_W-1, i*BIT_W);
        ap_uint<BIT_IN> temp_in = in((i+1)*BIT_IN-1, i*BIT_IN);
        ap_int<(BIT_W+BIT_IN)> result = temp_in * temp_w;
        accumulation += result;
    }
    return accumulation;
}

/*
*   output channel-wise parallel calculation
*   with relu activation (BN has been merged into conv layer)
*/
template <  int NUM_IN,
            int NUM_OUT,

            int BIT_IN,
            int FL_IN,
            int BIT_OUT,
            int FL_OUT,
            
            int BIT_W,
            int BIT_ALPHA,
            int FL_ALPHA,
            int BIT_BIAS,
            int BIT_TMP,
            
            int SIMD,
            int PE,
            int VECT_NUMS >
void conv3x3_relu(  hls::stream<ap_uint<BIT_IN*SIMD> >& in,
                    const ap_uint<BIT_W*SIMD> weights[PE][(NUM_IN/SIMD)*(NUM_OUT/PE)],
                    const ap_uint<BIT_ALPHA> alpha[PE][NUM_OUT/PE],
                    const ap_int<BIT_BIAS> bias[PE][NUM_OUT/PE],
                    hls::stream<ap_uint<BIT_OUT*PE> >& out,
                    const unsigned reps ){

#pragma HLS DATAFLOW

    const unsigned INPUT_FOLD = NUM_IN/SIMD;    /* input_channel / simd * kernel_size^2 */
	const unsigned OUTPUT_FOLD = NUM_OUT/PE;     /* output_channel / pe */
    const unsigned BIT_ACC = 18;

    const unsigned total_loop_num = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;

    ap_uint<BIT_IN*SIMD> input_temp_arr[INPUT_FOLD];
    #pragma HLS RESOURCE variable=input_temp_arr core=RAM_2P_BRAM

    unsigned in_fold_cnt = 0;
	unsigned out_fold_cnt = 0;
	unsigned tile = 0;

    ap_uint<BIT_IN*SIMD> input_temp;
    ap_int<BIT_ACC> acc[PE]; 
    ap_int<BIT_TMP> output_temp[PE];   
    ap_uint<BIT_OUT> output_uint[PE];

    for(int i=0; i<total_loop_num; i++){
    #pragma HLS PIPELINE II=1

        if(out_fold_cnt == 0){
            input_temp = in.read();
            input_temp_arr[in_fold_cnt] = input_temp;
        }
        else{
            input_temp = input_temp_arr[in_fold_cnt];
        }

        if(in_fold_cnt == 0){
            for(int j=0; j<PE; j++){
            #pragma HLS UNROLL

                acc[j] = 0;
            }
        }

        for(int k=0; k<PE; k++){
        #pragma HLS UNROLL

            ap_uint<BIT_IN*SIMD> weight_temp = weights[k][tile];
            acc[k] += chi_vector_dot_product_unsigned<BIT_IN, BIT_W, BIT_ACC, SIMD>(input_temp, weight_temp);
        }

        tile++;
        // cout << "in_fold_cnt:" << in_fold_cnt << endl;
        if(++in_fold_cnt == INPUT_FOLD){
            in_fold_cnt = 0;
            ap_uint<BIT_OUT*PE> out_buf;
            for(int p=0; p<PE; p++){
            #pragma HLS UNROLL

                output_temp[p] = acc[p] * alpha[p][out_fold_cnt];
                output_temp[p] += bias[p][out_fold_cnt];
                output_temp[p] = (output_temp[p]>(ap_int<BIT_TMP>)0) ? output_temp[p] : (ap_int<BIT_TMP>)0;
                output_uint[p] = truncate_unsigned<(BIT_TMP-FL_IN-FL_ALPHA), (FL_IN+FL_ALPHA), (BIT_OUT-FL_OUT), FL_OUT>(output_temp[p]);
                out_buf((p+1)*BIT_OUT-1, p*BIT_OUT) = output_uint[p];
            }

            out.write(out_buf);

            if(++out_fold_cnt == OUTPUT_FOLD){
                out_fold_cnt = 0;
                tile=0;
            }
        }
    }
}

template <  int NUM_IN,
            int NUM_OUT,

            int BIT_IN,
            int FL_IN,
            int BIT_OUT,
            int FL_OUT,

            int BIT_W,
            int BIT_ALPHA,
            int FL_ALPHA,
            int BIT_BIAS,
            int BIT_TMP,

            int SIMD,
            int PE,
            int VECT_NUMS >
void conv3x3_relu_c20(  hls::stream<ap_uint<BIT_IN*SIMD> >& in,
                    const ap_uint<BIT_W*SIMD> weights[PE][(NUM_IN/SIMD)*(NUM_OUT/PE)],
                    const ap_uint<BIT_ALPHA> alpha[PE][NUM_OUT/PE],
                    const ap_int<BIT_BIAS> bias[PE][NUM_OUT/PE],
                    hls::stream<ap_uint<BIT_OUT*PE> >& out,
                    const unsigned reps ){

#pragma HLS DATAFLOW

    const unsigned INPUT_FOLD = NUM_IN/SIMD;    /* input_channel / simd * kernel_size^2 */
	const unsigned OUTPUT_FOLD = NUM_OUT/PE;     /* output_channel / pe */
    const unsigned BIT_ACC = 18;

    const unsigned total_loop_num = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;

    ap_uint<BIT_IN*SIMD> input_temp_arr[INPUT_FOLD];
    #pragma HLS RESOURCE variable=input_temp_arr core=RAM_2P_BRAM

    unsigned in_fold_cnt = 0;
	unsigned out_fold_cnt = 0;
	unsigned tile = 0;

    ap_uint<BIT_IN*SIMD> input_temp;
    ap_int<BIT_ACC> acc[PE];
    ap_int<BIT_TMP> output_temp[PE];
    ap_uint<BIT_OUT> output_uint[PE];

    for(int i=0; i<total_loop_num; i++){
    #pragma HLS PIPELINE II=1

        if(out_fold_cnt == 0){
            input_temp = in.read();
            input_temp_arr[in_fold_cnt] = input_temp;
        }
        else{
            input_temp = input_temp_arr[in_fold_cnt];
        }

        if(in_fold_cnt == 0){
            for(int j=0; j<PE; j++){
            #pragma HLS UNROLL

                acc[j] = 0;
            }
        }

        for(int k=0; k<PE; k++){
        #pragma HLS UNROLL

            ap_uint<BIT_IN*SIMD> weight_temp = weights[k][tile];
            acc[k] += chi_vector_dot_product<BIT_IN, BIT_W, BIT_ACC, SIMD>(input_temp, weight_temp);
        }

        tile++;
        // cout << "in_fold_cnt:" << in_fold_cnt << endl;
        if(++in_fold_cnt == INPUT_FOLD){
            in_fold_cnt = 0;
            ap_uint<BIT_OUT*PE> out_buf;
            for(int p=0; p<PE; p++){
            #pragma HLS UNROLL

                output_temp[p] = acc[p] * alpha[p][out_fold_cnt];
                output_temp[p] += bias[p][out_fold_cnt];
                output_temp[p] = (output_temp[p]>(ap_int<BIT_TMP>)0) ? output_temp[p] : (ap_int<BIT_TMP>)0;
                output_uint[p] = truncate_unsigned<(BIT_TMP-FL_IN-FL_ALPHA), (FL_IN+FL_ALPHA), (BIT_OUT-FL_OUT), FL_OUT>(output_temp[p]);
                out_buf((p+1)*BIT_OUT-1, p*BIT_OUT) = output_uint[p];
            }

            out.write(out_buf);

            if(++out_fold_cnt == OUTPUT_FOLD){
                out_fold_cnt = 0;
                tile=0;
            }
        }
    }
}

template <  int ROW_IN,
            int COL_IN,
            int CH_IN,
            int BIT_IN,
            int FL_IN,

            int CH_OUT,
            int BIT_OUT,
            int FL_OUT,
            
            int BIT_W,
            int BIT_ALPHA,
            int FL_ALPHA,
            int BIT_BIAS,
            int BIT_TMP,
            
            int SIMD,
            int PE>
void conv3x3_layer( hls::stream<ap_uint<BIT_IN*CH_IN> >& in,
                    const ap_uint<BIT_W*SIMD> weights[PE][((CH_IN*9)/SIMD)*(CH_OUT/PE)],
                    const ap_uint<BIT_ALPHA> alpha[PE][CH_OUT/PE],
                    const ap_int<BIT_BIAS> bias[PE][CH_OUT/PE],
                    hls::stream<ap_uint<BIT_OUT*CH_OUT> >& out,
                    const unsigned reps ){

#pragma HLS DATAFLOW

    const unsigned INTER_ROW = ROW_IN + 2;	
	const unsigned INTER_COL = COL_IN + 2;
    const unsigned ROW_OUT = ROW_IN;
	const unsigned COL_OUT = COL_IN;

    stream<ap_uint<CH_IN*BIT_IN> > padding_out("padding_out");
	padding<ROW_IN, COL_IN, CH_IN, BIT_IN, 1>(in, padding_out, reps);

	stream<ap_uint<CH_IN*BIT_IN> > swu_out("swu_out");
	SWU<3, 1, INTER_ROW, INTER_COL, CH_IN, BIT_IN> (padding_out, swu_out, reps);

    stream<ap_uint<SIMD*BIT_IN> > adj_out("adj_out");
	StreamingDataWidthConverter_Batch<CH_IN*BIT_IN, SIMD*BIT_IN, 9*ROW_OUT*COL_OUT>(swu_out, adj_out, reps);

    hls::stream<ap_uint<BIT_OUT*PE> > conv_out("conv_out");

    conv3x3_relu<CH_IN*3*3, CH_OUT, BIT_IN, FL_IN, BIT_OUT, FL_OUT, BIT_W, BIT_ALPHA, FL_ALPHA, BIT_BIAS, BIT_TMP, SIMD, PE, ROW_OUT*COL_OUT>(
        adj_out, weights, alpha, bias, conv_out, reps );

    StreamingDataWidthConverter_Batch<PE*BIT_OUT, CH_OUT*BIT_OUT, ROW_OUT * COL_OUT * CH_OUT / PE>(
        conv_out, out, reps );
}

template <  int ROW_IN,
            int COL_IN,
            int CH_IN,
            int BIT_IN,
            int FL_IN,

            int CH_OUT,
            int BIT_OUT,
            int FL_OUT,
            
            int BIT_W,
            int BIT_ALPHA,
            int FL_ALPHA,
            int BIT_BIAS,
            int BIT_TMP,
            
            int SIMD,
            int PE>
void conv3x3_layer_c20( hls::stream<ap_uint<BIT_IN*CH_IN> >& in,
                    const ap_uint<BIT_W*SIMD> weights[PE][((CH_IN*9)/SIMD)*(CH_OUT/PE)],
                    const ap_uint<BIT_ALPHA> alpha[PE][CH_OUT/PE],
                    const ap_int<BIT_BIAS> bias[PE][CH_OUT/PE],
                    hls::stream<ap_uint<BIT_OUT*CH_OUT> >& out,
                    const unsigned reps ){

#pragma HLS DATAFLOW

    const unsigned INTER_ROW = ROW_IN + 2;
	const unsigned INTER_COL = COL_IN + 2;
    const unsigned ROW_OUT = ROW_IN;
	const unsigned COL_OUT = COL_IN;

    stream<ap_uint<CH_IN*BIT_IN> > padding_out("padding_out");
	padding<ROW_IN, COL_IN, CH_IN, BIT_IN, 1>(in, padding_out, reps);

	stream<ap_uint<CH_IN*BIT_IN> > swu_out("swu_out");
	SWU<3, 1, INTER_ROW, INTER_COL, CH_IN, BIT_IN> (padding_out, swu_out, reps);

    stream<ap_uint<SIMD*BIT_IN> > adj_out("adj_out");
	StreamingDataWidthConverter_Batch<CH_IN*BIT_IN, SIMD*BIT_IN, 9*ROW_OUT*COL_OUT>(swu_out, adj_out, reps);

    hls::stream<ap_uint<BIT_OUT*PE> > conv_out("conv_out");

    conv3x3_relu_c20<CH_IN*3*3, CH_OUT, BIT_IN, FL_IN, BIT_OUT, FL_OUT, BIT_W, BIT_ALPHA, FL_ALPHA, BIT_BIAS, BIT_TMP, SIMD, PE, ROW_OUT*COL_OUT>(
        adj_out, weights, alpha, bias, conv_out, reps );

    StreamingDataWidthConverter_Batch<PE*BIT_OUT, CH_OUT*BIT_OUT, ROW_OUT * COL_OUT * CH_OUT / PE>(
        conv_out, out, reps );
}

template <  int NUM_IN,
            int NUM_OUT,

            int BIT_IN,
            int FL_IN,
            int BIT_OUT,
            int FL_OUT,
            
            int BIT_W,
            int BIT_ALPHA,
            int FL_ALPHA,
            int BIT_BIAS,
            int BIT_TMP,
            
            int SIMD,
            int PE,
            int VECT_NUMS >
void conv3x3(   hls::stream<ap_uint<BIT_IN*SIMD> >& in,
                const ap_uint<BIT_W*SIMD> weights[PE][(NUM_IN/SIMD)*(NUM_OUT/PE)],
                const ap_uint<BIT_ALPHA> alpha[PE][NUM_OUT/PE],
                const ap_int<BIT_BIAS> bias[PE][NUM_OUT/PE],
                hls::stream<ap_uint<BIT_OUT*PE> >& out,
                const unsigned reps ){

#pragma HLS DATAFLOW

    const unsigned INPUT_FOLD = NUM_IN/SIMD;    /* input_channel / simd * kernel_size^2 */
	const unsigned OUTPUT_FOLD = NUM_OUT/PE;     /* output_channel / pe */
    const unsigned BIT_ACC = 18;

    const unsigned total_loop_num = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;

    ap_uint<BIT_IN*SIMD> input_temp_arr[INPUT_FOLD];
    #pragma HLS RESOURCE variable=input_temp_arr core=RAM_2P_BRAM

    unsigned in_fold_cnt = 0;
	unsigned out_fold_cnt = 0;
	unsigned tile = 0;

    ap_uint<BIT_IN*SIMD> input_temp;
    ap_int<BIT_ACC> acc[PE]; 
    ap_int<BIT_TMP> output_temp[PE];   

    for(int i=0; i<total_loop_num; i++){
    #pragma HLS PIPELINE II=1

        if(out_fold_cnt == 0){
            input_temp = in.read();
            input_temp_arr[in_fold_cnt] = input_temp;
        }
        else{
            input_temp = input_temp_arr[in_fold_cnt];
        }

        if(in_fold_cnt == 0){
            for(int j=0; j<PE; j++){
            #pragma HLS UNROLL

                acc[j] = 0;
            }
        }

        for(int k=0; k<PE; k++){
        #pragma HLS UNROLL

            ap_uint<BIT_IN*SIMD> weight_temp = weights[k][tile];
            acc[k] += chi_vector_dot_product_unsigned<BIT_IN, BIT_W, BIT_ACC, SIMD>(input_temp, weight_temp);
        }

        tile++;
        // cout << "in_fold_cnt:" << in_fold_cnt << endl;
        if(++in_fold_cnt == INPUT_FOLD){
            in_fold_cnt = 0;
            ap_uint<BIT_OUT*PE> out_buf;
            for(int p=0; p<PE; p++){
            #pragma HLS UNROLL

                output_temp[p] = acc[p] * alpha[p][out_fold_cnt];
                output_temp[p] += bias[p][out_fold_cnt];
                out_buf((p+1)*BIT_OUT-1, p*BIT_OUT) = output_temp[p];
            }

            out.write(out_buf);

            if(++out_fold_cnt == OUTPUT_FOLD){
                out_fold_cnt = 0;
                tile=0;
            }
        }
    }
}

template <  int ROW_IN,
            int COL_IN,
            int CH_IN,
            int BIT_IN,
            int FL_IN,

            int CH_OUT,
            int BIT_OUT,
            int FL_OUT,
            
            int BIT_W,
            int BIT_ALPHA,
            int FL_ALPHA,
            int BIT_BIAS,
            int BIT_TMP,
            
            int SIMD,
            int PE>
void conv3x3_layer_crc( hls::stream<ap_uint<BIT_IN*CH_IN> >& in,
                    const ap_uint<BIT_W*SIMD> weights[PE][((CH_IN*9)/SIMD)*(CH_OUT/PE)],
                    const ap_uint<BIT_ALPHA> alpha[PE][CH_OUT/PE],
                    const ap_int<BIT_BIAS> bias[PE][CH_OUT/PE],
                    hls::stream<ap_uint<BIT_OUT*PE> >& out,
                    const unsigned reps ){

#pragma HLS DATAFLOW

    const unsigned INTER_ROW = ROW_IN + 2;	
	const unsigned INTER_COL = COL_IN + 2;
    const unsigned ROW_OUT = ROW_IN;
	const unsigned COL_OUT = COL_IN;

    stream<ap_uint<CH_IN*BIT_IN> > padding_out("padding_out");
	padding<ROW_IN, COL_IN, CH_IN, BIT_IN, 1>(in, padding_out, reps);

	stream<ap_uint<CH_IN*BIT_IN> > swu_out("swu_out");
	SWU<3, 1, INTER_ROW, INTER_COL, CH_IN, BIT_IN> (padding_out, swu_out, reps);

    stream<ap_uint<SIMD*BIT_IN> > adj_out("adj_out");
	StreamingDataWidthConverter_Batch<CH_IN*BIT_IN, SIMD*BIT_IN, 9*ROW_OUT*COL_OUT>(swu_out, adj_out, reps);

    // hls::stream<ap_uint<BIT_OUT*PE> > conv_out("conv_out");

    conv3x3<CH_IN*3*3, CH_OUT, BIT_IN, FL_IN, BIT_OUT, FL_OUT, BIT_W, BIT_ALPHA, FL_ALPHA, BIT_BIAS, BIT_TMP, SIMD, PE, ROW_OUT*COL_OUT>(
        adj_out, weights, alpha, bias, out, reps );

    // StreamingDataWidthConverter_Batch<PE*BIT_OUT, CH_OUT*BIT_OUT, ROW_OUT * COL_OUT * CH_OUT / PE>(
    //     conv_out, out, reps );
}

void atss_0426_round(   hls::stream<my_ap_axis >& in,
                        hls::stream<my_ap_axis >& out,
                        const unsigned int reps ){

#pragma HLS DATAFLOW
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=reps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable = backbone_C2_0_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C2_0_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C2_0_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = backbone_C2_2_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C2_2_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C2_2_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = backbone_C2_4_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C2_4_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C2_4_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = backbone_C3_1_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C3_1_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C3_1_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = backbone_C4_1_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C4_1_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C4_1_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = backbone_C4_2_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C4_2_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C4_2_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = backbone_C4_3_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C4_3_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C4_3_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = backbone_C4_4_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C4_4_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = backbone_C4_4_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = cls_0_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = cls_0_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = cls_0_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = cls_1_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = cls_1_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = cls_1_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = cls_2_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = cls_2_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = cls_2_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = cls_3_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = cls_3_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = cls_3_conv_bias_q complete dim = 1

#pragma HLS ARRAY_PARTITION variable = last_conv_weight_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = last_conv_alpha_q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = last_conv_bias_q complete dim = 1

    //  const unsigned int reps = 1;

    const unsigned int num_per_rep = 360 * 640 * 3 * 8 / 64;

    hls::stream<ap_uint<64> > in_stream_extract("in_stream_extract");
    #pragma HLS STREAM variable=in_stream_extract depth=16 dim=1
	ExtractPixels<64, num_per_rep> (in, in_stream_extract, reps);

    hls::stream<ap_uint<64 * 3> > in_stream0("in_stream0");
    #pragma HLS STREAM variable=in_stream0 depth=16 dim=1
    StreamingDataWidthConverter_Batch<64, 64 * 3, num_per_rep>(in_stream_extract, in_stream0, reps);

    hls::stream<ap_uint<8 * 3> > in_stream1("in_stream1");
    #pragma HLS STREAM variable=in_stream1 depth=16 dim=1
	StreamingDataWidthConverter_Batch<64 * 3, 8 * 3, num_per_rep / 3> (in_stream0, in_stream1, reps);

    hls::stream<ap_uint<8 * 3> > in_stream2("in_stream2");
    #pragma HLS STREAM variable=in_stream2 depth=16 dim=1

    resize_batch(in_stream1, in_stream2, reps);

    #ifdef DEBUG
    cout << "in_stream2 size:" << in_stream2.size() << endl;
    #endif

    hls::stream<ap_uint<8 * 3> > in_stream3("in_stream3");
    #pragma HLS STREAM variable=in_stream3 depth=16 dim=1

    img_norm<8, 3, 192, 352>(in_stream2, in_stream3, reps);

    #ifdef DEBUG
    cout << "in_stream3 size:" << in_stream3.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 16> > conv_c20_out("conv_c20_out");
    #pragma HLS STREAM variable=conv_c20_out depth=256 dim=1
    conv3x3_layer_c20<192, 352, 3, 8, 3, 16, 5, 0, 4, 15, 16, 21, 30, 3, 16>(
        in_stream3, backbone_C2_0_conv_weight_q, backbone_C2_0_conv_alpha_q, backbone_C2_0_conv_bias_q, conv_c20_out, reps );

    #ifdef DEBUG
    cout << "conv_c20_out size:" << conv_c20_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 16> > pool_c20_out("pool_c20_out");
    #pragma HLS STREAM variable=pool_c20_out depth=256 dim=1
    max_pool2d<2, 192, 352, 16, 5>(conv_c20_out, pool_c20_out, reps);

    #ifdef DEBUG
    cout << "pool_c20_out size:" << pool_c20_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 32> > conv_c22_out("conv_c22_out");
    #pragma HLS STREAM variable=conv_c22_out depth=256 dim=1

    conv3x3_layer<96, 176, 16, 5, 0, 32, 5, 0, 4, 15, 19, 22, 30, 16, 8>(
        pool_c20_out, backbone_C2_2_conv_weight_q, backbone_C2_2_conv_alpha_q, backbone_C2_2_conv_bias_q, conv_c22_out, reps );
    
    #ifdef DEBUG
    cout << "conv_c22_out size:" << conv_c22_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 32> > pool_c22_out("pool_c22_out");
    #pragma HLS STREAM variable=pool_c22_out depth=256 dim=1
    max_pool2d<2, 96, 176, 32, 5>(conv_c22_out, pool_c22_out, reps);

    #ifdef DEBUG
    cout << "pool_c22_out size:" << pool_c22_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 64> > conv_c24_out("conv_c24_out");
    #pragma HLS STREAM variable=conv_c24_out depth=256 dim=1

    conv3x3_layer<48, 88, 32, 5, 0, 64, 5, 1, 4, 15, 19, 21, 30, 32, 4>(
        pool_c22_out, backbone_C2_4_conv_weight_q, backbone_C2_4_conv_alpha_q, backbone_C2_4_conv_bias_q, conv_c24_out, reps );
    
    #ifdef DEBUG
    cout << "conv_c24_out size:" << conv_c24_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 64> > pool_c24_out("pool_c24_out");
    #pragma HLS STREAM variable=pool_c24_out depth=256 dim=1
    max_pool2d<2, 48, 88, 64, 5>(conv_c24_out, pool_c24_out, reps);

    #ifdef DEBUG
    cout << "pool_c24_out size:" << pool_c24_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 64> > conv_c31_out("conv_c31_out");
    #pragma HLS STREAM variable=conv_c31_out depth=256 dim=1

    conv3x3_layer<24, 44, 64, 5, 1, 64, 5, 1, 4, 15, 19, 22, 30, 64, 1>(
        pool_c24_out, backbone_C3_1_conv_weight_q, backbone_C3_1_conv_alpha_q, backbone_C3_1_conv_bias_q, conv_c31_out, reps );
    
    #ifdef DEBUG
    cout << "conv_c31_out size:" << conv_c31_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 64> > pool_c31_out("pool_c31_out");
    #pragma HLS STREAM variable=pool_c31_out depth=256 dim=1
    max_pool2d<2, 24, 44, 64, 5>(conv_c31_out, pool_c31_out, reps);

    #ifdef DEBUG
    cout << "pool_c31_out size:" << pool_c31_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 64> > conv_c41_out("conv_c41_out");
    #pragma HLS STREAM variable=conv_c41_out depth=256 dim=1

    conv3x3_layer<12, 22, 64, 5, 1, 64, 5, 2, 4, 15, 20, 22, 30, 16, 1>(
        pool_c31_out, backbone_C4_1_conv_weight_q, backbone_C4_1_conv_alpha_q, backbone_C4_1_conv_bias_q, conv_c41_out, reps );
    
    #ifdef DEBUG
    cout << "conv_c41_out size:" << conv_c41_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 64> > conv_c42_out("conv_c42_out");
    #pragma HLS STREAM variable=conv_c42_out depth=256 dim=1

    conv3x3_layer<12, 22, 64, 5, 2, 64, 5, 2, 4, 15, 19, 23, 30, 16, 1>(
        conv_c41_out, backbone_C4_2_conv_weight_q, backbone_C4_2_conv_alpha_q, backbone_C4_2_conv_bias_q, conv_c42_out, reps );
    
    #ifdef DEBUG
    cout << "conv_c42_out size:" << conv_c42_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 64> > conv_c43_out("conv_c43_out");
    #pragma HLS STREAM variable=conv_c43_out depth=256 dim=1

    conv3x3_layer<12, 22, 64, 5, 2, 64, 5, 2, 4, 15, 19, 22, 30, 16, 1>(
        conv_c42_out, backbone_C4_3_conv_weight_q, backbone_C4_3_conv_alpha_q, backbone_C4_3_conv_bias_q, conv_c43_out, reps );
    
    #ifdef DEBUG
    cout << "conv_c43_out size:" << conv_c43_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 32> > conv_c44_out("conv_c44_out");
    #pragma HLS STREAM variable=conv_c44_out depth=256 dim=1

    conv3x3_layer<12, 22, 64, 5, 2, 32, 5, 1, 4, 15, 19, 22, 30, 8, 1>(
        conv_c43_out, backbone_C4_4_conv_weight_q, backbone_C4_4_conv_alpha_q, backbone_C4_4_conv_bias_q, conv_c44_out, reps );
    
    #ifdef DEBUG
    cout << "conv_c44_out size:" << conv_c44_out.size() << endl;
    #endif

    /* cls_reg_centerness */

    hls::stream<ap_uint<5 * 32> > cls_0_conv_out("cls_0_conv_out");
    #pragma HLS STREAM variable=cls_0_conv_out depth=256 dim=1

    conv3x3_layer<12, 22, 32, 5, 1, 32, 5, 1, 4, 15, 19, 22, 30, 4, 1>(
        conv_c44_out, cls_0_conv_weight_q, cls_0_conv_alpha_q, cls_0_conv_bias_q, cls_0_conv_out, reps );
    
    #ifdef DEBUG
    cout << "cls_0_conv_out size:" << cls_0_conv_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 32> > cls_1_conv_out("cls_1_conv_out");
    #pragma HLS STREAM variable=cls_1_conv_out depth=256 dim=1

    conv3x3_layer<12, 22, 32, 5, 1, 32, 5, 2, 4, 15, 19, 22, 30, 4, 1>(
        cls_0_conv_out, cls_1_conv_weight_q, cls_1_conv_alpha_q, cls_1_conv_bias_q, cls_1_conv_out, reps );
    
    #ifdef DEBUG
    cout << "cls_1_conv_out size:" << cls_1_conv_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 32> > cls_2_conv_out("cls_2_conv_out");
    #pragma HLS STREAM variable=cls_2_conv_out depth=256 dim=1

    conv3x3_layer<12, 22, 32, 5, 2, 32, 5, 2, 4, 15, 19, 24, 30, 4, 1>(
        cls_1_conv_out, cls_2_conv_weight_q, cls_2_conv_alpha_q, cls_2_conv_bias_q, cls_2_conv_out, reps );
    
    #ifdef DEBUG
    cout << "cls_2_conv_out size:" << cls_2_conv_out.size() << endl;
    #endif

    hls::stream<ap_uint<5 * 32> > cls_3_conv_out("cls_3_conv_out");
    #pragma HLS STREAM variable=cls_3_conv_out depth=256 dim=1

    conv3x3_layer<12, 22, 32, 5, 2, 32, 5, 3, 4, 15, 19, 23, 30, 4, 1>(
        cls_2_conv_out, cls_3_conv_weight_q, cls_3_conv_alpha_q, cls_3_conv_bias_q, cls_3_conv_out, reps );
    
    #ifdef DEBUG
    cout << "cls_3_conv_out size:" << cls_3_conv_out.size() << endl;
    #endif

    hls::stream<ap_uint<1 * 32> > last_conv_out("last_conv_out");
    #pragma HLS STREAM variable=last_conv_out depth=256 dim=1

    conv3x3_layer_crc<12, 22, 32, 5, 3, 17, 32, 20, 4, 15, 17, 22, 32, 2, 1>(
        cls_3_conv_out, last_conv_weight_q, last_conv_alpha_q, last_conv_bias_q, last_conv_out, reps );
    
    #ifdef DEBUG
    cout << "last_conv_out size:" << last_conv_out.size() << endl;
    #endif

    // hls::stream<ap_uint<64 * 3> > in_stream4("in_stream4");
    // #pragma HLS STREAM variable=in_stream4 depth=16 dim=1
    // StreamingDataWidthConverter_Batch<8 * 3, 64 * 3, 192*352> (in_stream3, in_stream4, reps);

    // hls::stream<ap_uint<5 * 1> > s0("s0");
    // #pragma HLS STREAM variable=s0 depth=128 dim=1
    // StreamingDataWidthConverter_Batch<5*32, 5*1, 12*22>(cls_3_conv_out, s0, reps);

    // hls::stream<ap_uint<8 * 1> > s1("s1");
    // #pragma HLS STREAM variable=s1 depth=128 dim=1
    // AppendZeros<5, 8, 12*22*32>(s0, s1, reps);

    // hls::stream<ap_uint<8 * 32> > s2("s2");
    // #pragma HLS STREAM variable=s2 depth=128 dim=1
    // StreamingDataWidthConverter_Batch<8*1, 8*32, 12*22*32>(s1, s2, reps);    
    
    hls::stream<ap_uint<64> >  conv_out("conv_out");
    #pragma HLS STREAM variable=conv_out depth=256 dim=1
    StreamingDataWidthConverter_Batch<1*32, 64, 12*22*17> (last_conv_out, conv_out, reps);
    AddLast<12*22*17*32/64>(conv_out, out, reps);

}


#ifdef DEBUG

#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>
#include <fstream>



void load_data(const char *path, char *ptr, unsigned int size)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f)
    {
        std::cout << "no such file,please check the file name!/n";
        exit(0);
    }
    f.read(ptr, size);
    f.close();
}

void write_data(const char *path, char *ptr, unsigned int size)
{
    std::ofstream f(path, std::ios::out | std::ios::binary);
    if (!f)
    {
        std::cout << "write no such file,please check the file name!/n";
        exit(0);
    }
    f.write(ptr, size);
    f.close();
}

int main(int argc, char const *argv[]){

    uint8_t img[360][640][3];
    load_data("C:\\RC4ML\\DAC\\ZJU2021\\HLS\\atss_0426_round\\img\\0142.bin", (char *) img, sizeof(img));

    const int data_points_per_line = 8;
    const int nums_line_pre_img = 360 * 640 * 3 * 8 / 64;
    uint8_t * data = (uint8_t *) img;

//    for(int j=0; j<360; j++){
//        for(int k=0; k<640; k++){
//            for(int i=0; i<3; i++){
//                printf(" %3d", img[j][k][i]);
//            }
//            cout << endl;
//        }
//    }

    hls::stream<my_ap_axis > input_stream("input stream");

	for (unsigned int i = 0; i < nums_line_pre_img; i++) {
	 	my_ap_axis temp;
	 	for (unsigned int j = 0; j < data_points_per_line; j++) {
	 		temp.data( 8*(j+1)-1, 8*j ) = data[i * data_points_per_line + j];
	 	}
	 	input_stream.write(temp);
	}

    cout << "input size :" << input_stream.size() << endl;
    cout << "start ..... " << endl;

    hls::stream<my_ap_axis > out_stream("out_stream");
    // hls::stream<my_ap_axis > out_stream_cls("out_stream_cls");
    // hls::stream<my_ap_axis > out_stream_reg("out_stream_reg");

    atss_0426_round(input_stream, out_stream, 1);


    while(!out_stream.empty()){
        static uint8_t flag = 0;
        my_ap_axis out_read = out_stream.read();
        for(int i=0; i<2; i++){
            ap_int<32> value_tmp = out_read.data(32*(i+1)-1, 32*i);
            // ap_fixed<32, 12, AP_RND> value = *(ap_fixed<32, 12, AP_RND>*)&value_tmp;
            // ap_fixed<8, 4, AP_RND> value = *(ap_fixed<8, 4, AP_RND>*)&value_tmp;
            cout << value_tmp << " " ;
            if(++flag == 17){
                flag = 0;
                cout << endl;
            }
        }
        // if(++flag==2){
        //     flag = 0;
        //     cout << endl;
        // }
    }

   return 0;
}

// int main(){

// 	// ap_fixed<8, 5, AP_RND> t_max = pow(2, 4) - 1.0/pow(2, 3);
// 	// ap_fixed<8, 5, AP_RND> t_min = -pow(2, 4);

// 	// cout << "t_max: " << t_max << endl;
// 	// cout << "t_min: " << t_min << endl;

//     ap_uint<4> test = (1<<4) - 1;

//     cout << test << endl;

// 	return 0;
// }

#endif
