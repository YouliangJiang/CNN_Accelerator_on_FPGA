#pragma once

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
#include "stream_tools.h"



template <	unsigned K,
			unsigned S,
			unsigned Din_H,
			unsigned Din_W,
			unsigned Cin,
			unsigned Ibit>
void SWU(
	stream<ap_uint<Cin*Ibit> >& in, 
	stream<ap_uint<Cin*Ibit> >& out, 
	const unsigned reps) 
{

	const unsigned steps = (Din_W-K)/S+1;
	const unsigned line_buffer_size = K*Din_W;
#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS RESOURCE variable line_buffer core=RAM_2P
// #pragma HLS RESOURCE variable line_buffer core=RAM_2P_LUTRAM

	ap_uint<Cin*Ibit> temp_in;

	ap_uint<1> initial_fill = 0;
	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;

	for (unsigned rep = 0; rep < reps*Din_H; rep++) {

		if (h == Din_H) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0; w < Din_W; w++) {
#pragma HLS PIPELINE II=1
			temp_in = in.read();
			
			unsigned line_buffer_pointer = pointer + w;
			if (line_buffer_pointer >= line_buffer_size) {
				line_buffer_pointer = line_buffer_pointer - line_buffer_size;
			}
#ifdef SWU_DEBUG
			cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
#endif
			line_buffer[line_buffer_pointer] = temp_in;
		}

		stride += 1;
		pointer += Din_W;
		if (pointer >= line_buffer_size) {
			pointer = pointer - line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < line_buffer_size; j++) {
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;

			unsigned s = 0;
			unsigned x = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*(K*K); i++ ) {
#pragma HLS PIPELINE II=1
				unsigned read_address = (pointer+s*S) + y*Din_W + x;

				if (read_address >= line_buffer_size)
					read_address = read_address - line_buffer_size;
#ifdef SWU_DEBUG
				cout << "read_address: " << read_address << endl;
#endif
				ap_uint<Cin*Ibit> temp_out = line_buffer[read_address];
				out.write(temp_out);

				if (x == K-1) {
					x = 0;
					if (y == K-1) {
						y = 0;
						if (s == steps-1)
							s = 0;
						else
							s++;
					}
					else
						y++;
				}
				else
					x++;
			}
		}
	}
}


template <	unsigned K,
			unsigned S,
			unsigned Din_H,
			unsigned Din_W,
			unsigned Cin,
			unsigned Ibit,
			unsigned SIMD>
void SWU_c24_pool(
	stream<ap_uint<SIMD*Ibit> >& in, 
	stream<ap_uint<SIMD*Ibit> >& out, 
	const unsigned reps) 
{

	const unsigned steps = (Din_W-K)/S+1;
	const unsigned line_buffer_size = K*Din_W;
#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS RESOURCE variable line_buffer core=RAM_2P
// #pragma HLS RESOURCE variable line_buffer core=RAM_2P_LUTRAM

	ap_uint<Cin*Ibit> temp_in = 0;
	ap_uint<Cin*Ibit> temp_out = 0;
	ap_uint<1> initial_fill = 0;

	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;
	unsigned cnt_in = 0;
	unsigned cnt_out = 0;

	for (unsigned rep = 0; rep < reps*Din_H; rep++) {

		if (h == Din_H) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0, o = 0; w < Din_W; w++) {
		#pragma HLS PIPELINE II=1

			for(unsigned tile_in=0; tile_in<(Cin/SIMD); tile_in++){
				temp_in((tile_in+1)*SIMD*Ibit-1, tile_in*Ibit*SIMD) = in.read();
			}
			// ap_uint<SIMD*Ibit> tmp_in = in.read();
			// temp_in = temp_in >> (SIMD*Ibit);
			// temp_in(Cin*Ibit-1, Cin*Ibit-SIMD*Ibit) = tmp_in;
			// if(++cnt_in == (Cin/SIMD)){
			// 	cnt_in = 0;

			// 	unsigned line_buffer_pointer = pointer + o;
			// 	if (line_buffer_pointer >= line_buffer_size) {
			// 		line_buffer_pointer = line_buffer_pointer - line_buffer_size;
			// 	}
			// 	#ifdef SWU_DEBUG
			// 	cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
			// 	#endif
			// 	line_buffer[line_buffer_pointer] = temp_in;

			// 	o++;
			// }
		}

		stride += 1;
		pointer += Din_W;
		if (pointer >= line_buffer_size) {
			pointer = pointer - line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < line_buffer_size; j++) {
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;

			unsigned s = 0;
			unsigned x = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*(K*K); i++ ) {
			#pragma HLS PIPELINE II=1

				if(cnt_out == 0){

					unsigned read_address = (pointer+s*S) + y*Din_W + x;

					if (read_address >= line_buffer_size)
						read_address = read_address - line_buffer_size;

					#ifdef SWU_DEBUG
					cout << "read_address: " << read_address << endl;
					#endif

					temp_out = line_buffer[read_address];

					if (x == K-1) {
						x = 0;
						if (y == K-1) {
							y = 0;
							if (s == steps-1)
								s = 0;
							else
								s++;
						}
						else
							y++;
					}
					else
						x++;
				}
				for(unsigned tile_out=0; tile_out<(Cin/SIMD); tile_out++){
					out.write(temp_out((tile_out+1)*Ibit*SIMD-1, tile_out*Ibit*SIMD));
				}
				// ap_uint<SIMD*Ibit> tmp_out = temp_out(SIMD*Ibit-1, 0);
				// out.write(tmp_out);
				// temp_out = temp_out >> (SIMD*Ibit);
				// if(++cnt_out == (Cin/SIMD)){
				// 	cnt_out = 0;
				// }
			}
		}
	}
}

template <	unsigned K,
			unsigned S,
			unsigned Din_H,
			unsigned Din_W,
			unsigned Cin,
			unsigned Ibit,
			unsigned SIMD>
void SWU_c42_pool(
	stream<ap_uint<SIMD*Ibit> >& in, 
	stream<ap_uint<SIMD*Ibit> >& out, 
	const unsigned reps) 
{

	const unsigned steps = (Din_W-K)/S+1;
	const unsigned line_buffer_size = K*Din_W;
#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS RESOURCE variable line_buffer core=RAM_2P
// #pragma HLS RESOURCE variable line_buffer core=RAM_2P_LUTRAM

	ap_uint<Cin*Ibit> temp_in = 0;
	ap_uint<Cin*Ibit> temp_out = 0;
	ap_uint<1> initial_fill = 0;

	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;
	unsigned cnt_in = 0;
	unsigned cnt_out = 0;

	for (unsigned rep = 0; rep < reps*Din_H; rep++) {

		if (h == Din_H) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0, o = 0; w < Din_W*Cin/SIMD; w++) {
		#pragma HLS PIPELINE II=1

			// for(unsigned tile_in=0; tile_in<(Cin/SIMD); tile_in++){
			// 	temp_in((tile_in+1)*SIMD*Ibit-1, tile_in*Ibit*SIMD) = in.read();
			// }
			ap_uint<SIMD*Ibit> tmp_in = in.read();
			temp_in = temp_in >> (SIMD*Ibit);
			temp_in(Cin*Ibit-1, Cin*Ibit-SIMD*Ibit) = tmp_in;
			if(++cnt_in == (Cin/SIMD)){
				cnt_in = 0;

				unsigned line_buffer_pointer = pointer + o;
				if (line_buffer_pointer >= line_buffer_size) {
					line_buffer_pointer = line_buffer_pointer - line_buffer_size;
				}
				#ifdef SWU_DEBUG
				cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
				#endif
				line_buffer[line_buffer_pointer] = temp_in;

				o++;
			}
		}

		stride += 1;
		pointer += Din_W;
		if (pointer >= line_buffer_size) {
			pointer = pointer - line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < line_buffer_size; j++) {
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;

			unsigned s = 0;
			unsigned x = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*(K*K)*Cin/SIMD; i++ ) {
			#pragma HLS PIPELINE II=1

				if(cnt_out == 0){

					unsigned read_address = (pointer+s*S) + y*Din_W + x;

					if (read_address >= line_buffer_size)
						read_address = read_address - line_buffer_size;

					#ifdef SWU_DEBUG
					cout << "read_address: " << read_address << endl;
					#endif

					temp_out = line_buffer[read_address];

					if (x == K-1) {
						x = 0;
						if (y == K-1) {
							y = 0;
							if (s == steps-1)
								s = 0;
							else
								s++;
						}
						else
							y++;
					}
					else
						x++;
				}
				for(unsigned tile_out=0; tile_out<(Cin/SIMD); tile_out++){
					out.write(temp_out((tile_out+1)*Ibit*SIMD-1, tile_out*Ibit*SIMD));
				}
				// ap_uint<SIMD*Ibit> tmp_out = temp_out(SIMD*Ibit-1, 0);
				// out.write(tmp_out);
				// temp_out = temp_out >> (SIMD*Ibit);
				// if(++cnt_out == (Cin/SIMD)){
				// 	cnt_out = 0;
				// }
			}
		}
	}
}

