#ifndef NVJPEGDECODER_SAMPLE_DECODER_H
#define NVJPEGDECODER_SAMPLE_DECODER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/time.h>  // timings
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
        {                                                                       \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }

#define CHECK_NVJPEG(call)                                                      \
    {                                                                           \
        nvjpegStatus_t _e = (call);                                             \
        if (_e != NVJPEG_STATUS_SUCCESS)                                        \
        {                                                                       \
            std::cout << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }


int dev_malloc(void **p, size_t s) { return (int) cudaMalloc(p, s); }

int dev_free(void *p) { return (int) cudaFree(p); }

int host_malloc(void **p, size_t s, unsigned int f) { return (int) cudaHostAlloc(p, s, f); }

int host_free(void *p) { return (int) cudaFreeHost(p); }

struct decode_params_t {
    int batch_size;
    int dev;
    int warmup;

    nvjpegJpegState_t nvjpeg_state;
    nvjpegHandle_t nvjpeg_handle;
    cudaStream_t stream;

    // used with decoupled API
    nvjpegJpegState_t nvjpeg_decoupled_state;
    nvjpegBufferPinned_t pinned_buffers[2]; // 2 buffers for pipelining
    nvjpegBufferDevice_t device_buffer;
    nvjpegJpegStream_t jpeg_streams[2]; //  2 streams for pipelining
    nvjpegDecodeParams_t nvjpeg_decode_params;
    nvjpegJpegDecoder_t nvjpeg_decoder;

    nvjpegOutputFormat_t fmt;
    bool write_decoded;
    bool hw_decode_available;
};

void create_decoupled_api_handles(decode_params_t &params) {

    CHECK_NVJPEG(nvjpegDecoderCreate(params.nvjpeg_handle,
                                     NVJPEG_BACKEND_DEFAULT,
                                     &params.nvjpeg_decoder));
    CHECK_NVJPEG(nvjpegDecoderStateCreate(params.nvjpeg_handle,
                                          params.nvjpeg_decoder,
                                          &params.nvjpeg_decoupled_state));

    CHECK_NVJPEG(nvjpegBufferPinnedCreate(params.nvjpeg_handle,
                                          NULL,
                                          &params.pinned_buffers[0]));
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(params.nvjpeg_handle,
                                          NULL,
                                          &params.pinned_buffers[1]));
    CHECK_NVJPEG(nvjpegBufferDeviceCreate(params.nvjpeg_handle,
                                          NULL,
                                          &params.device_buffer));

    CHECK_NVJPEG(nvjpegJpegStreamCreate(params.nvjpeg_handle,
                                        &params.jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(params.nvjpeg_handle,
                                        &params.jpeg_streams[1]));

    CHECK_NVJPEG(nvjpegDecodeParamsCreate(params.nvjpeg_handle,
                                          &params.nvjpeg_decode_params));
}

void destroy_decoupled_api_handles(decode_params_t &params) {

    CHECK_NVJPEG(nvjpegDecodeParamsDestroy(params.nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegJpegStreamDestroy(params.jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegJpegStreamDestroy(params.jpeg_streams[1]));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(params.pinned_buffers[0]));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(params.pinned_buffers[1]));
    CHECK_NVJPEG(nvjpegBufferDeviceDestroy(params.device_buffer));
    CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_decoupled_state));
    CHECK_NVJPEG(nvjpegDecoderDestroy(params.nvjpeg_decoder));
}

#endif //NVJPEGDECODER_SAMPLE_DECODER_H
