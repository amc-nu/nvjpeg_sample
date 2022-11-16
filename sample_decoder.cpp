#include "sample_decoder.h"
#include "helper_timer.h"

bool read_image(const std::string &file_name, std::vector<char> &raw_data, size_t &raw_len)
{
  // Read an image from disk.
  std::ifstream input(file_name.c_str(),
                      std::ios::in | std::ios::binary | std::ios::ate);
  // Get the size
  std::streamsize file_size = input.tellg();
  input.seekg(0, std::ios::beg);
  // resize if buffer is too small
  if (raw_data.size() < file_size)
  {
    raw_data.resize(file_size);
  }
  if (!input.read(raw_data.data(), file_size))
  {
    std::cerr << "Cannot read from file: " << file_name << std::endl;
    return false;
  }
  raw_len = file_size;
  return true;
}

bool prepare_buffers(std::vector<char> &file_data,
                     size_t &file_len,
                     decode_params_t &params,
                     size_t &out_width,
                     size_t &out_height,
                     nvjpegImage_t &ibuf,
                     nvjpegImage_t &isz)
{
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;

  CHECK_NVJPEG(nvjpegGetImageInfo(
    params.nvjpeg_handle, (unsigned char *) file_data.data(), file_len,
    &channels, &subsampling, widths, heights));

  out_width = widths[0];
  out_height = heights[0];
  std::cout << "Image is " << channels << " channels." << std::endl;
  for (int c = 0; c < channels; c++)
  {
    std::cout << "Channel #" << c << " size: " << widths[c] << " x "
              << heights[c] << std::endl;
  }
  switch (subsampling)
  {
    case NVJPEG_CSS_444:
      std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
      break;
    case NVJPEG_CSS_440:
      std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
      break;
    case NVJPEG_CSS_422:
      std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
      break;
    case NVJPEG_CSS_420:
      std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
      break;
    case NVJPEG_CSS_411:
      std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
      break;
    case NVJPEG_CSS_410:
      std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
      break;
    case NVJPEG_CSS_GRAY:
      std::cout << "Grayscale JPEG " << std::endl;
      break;
    case NVJPEG_CSS_UNKNOWN:
      std::cout << "Unknown chroma subsampling" << std::endl;
      return false;
  }
  int mul = 1;
  // in the case of interleaved RGB output, write only to single channel, but
  // 3 samples at once
  if (params.fmt == NVJPEG_OUTPUT_RGBI || params.fmt == NVJPEG_OUTPUT_BGRI)
  {
    channels = 1;
    mul = 3;
  }
    // in the case of rgb create 3 buffers with sizes of original image
  else if (params.fmt == NVJPEG_OUTPUT_RGB ||
           params.fmt == NVJPEG_OUTPUT_BGR)
  {
    channels = 3;
    widths[1] = widths[2] = widths[0];
    heights[1] = heights[2] = heights[0];
  }
  // realloc output buffer if required
  for (int c = 0; c < channels; c++)
  {
    int aw = mul * widths[c];
    int ah = heights[c];
    int sz = aw * ah;
    ibuf.pitch[c] = aw;
    if (sz > isz.pitch[c])
    {
      if (ibuf.channel[c])
      {
        CHECK_CUDA(cudaFree(ibuf.channel[c]));
      }
      CHECK_CUDA(cudaMalloc(&ibuf.channel[c], sz));
      isz.pitch[c] = sz;
    }
  }
  return true;
}// prepare_buffers

bool decode_image(std::vector<char> &raw_data,
                  size_t &raw_len,
                  nvjpegImage_t &out_nvjpeg,
                  decode_params_t &params,
                  double &process_time)
{

  CHECK_CUDA(cudaStreamSynchronize(params.stream));
  cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
  float loopTime = 0;

  CHECK_CUDA(cudaEventCreate(&startEvent, cudaEventBlockingSync));
  CHECK_CUDA(cudaEventCreate(&stopEvent, cudaEventBlockingSync));

  std::vector<const unsigned char *> batched_bitstreams;
  std::vector<size_t> batched_bitstreams_size;
  std::vector<nvjpegImage_t> batched_output;

  // bit-streams that batched decode cannot handle
  std::vector<const unsigned char *> otherdecode_bitstreams;
  std::vector<size_t> otherdecode_bitstreams_size;
  std::vector<nvjpegImage_t> otherdecode_output;
#ifdef CUDA11
  if(params.hw_decode_available){
      // extract bitstream meta data to figure out whether a bit-stream can be decoded
      nvjpegJpegStreamParseHeader(params.nvjpeg_handle,
                                  (const unsigned char *)raw_data.data(),
                                  raw_len,
                                  params.jpeg_streams[0]);
      int isSupported = -1;
      nvjpegDecodeBatchedSupported(params.nvjpeg_handle,
                                   params.jpeg_streams[0],
                                   &isSupported);

      if(isSupported == 0){
          batched_bitstreams.push_back((const unsigned char *)raw_data.data());
          batched_bitstreams_size.push_back(raw_len);
          batched_output.push_back(out_nvjpeg);
      } else {
          otherdecode_bitstreams.push_back((const unsigned char *)raw_data.data());
          otherdecode_bitstreams_size.push_back(raw_len);
          otherdecode_output.push_back(out_nvjpeg);
      }
  } else
#endif
  {
    otherdecode_bitstreams.push_back((const unsigned char *) raw_data.data());
    otherdecode_bitstreams_size.push_back(raw_len);
    otherdecode_output.push_back(out_nvjpeg);
  }

  CHECK_CUDA(cudaEventRecord(startEvent, params.stream));

  if (!batched_bitstreams.empty())
  {
    CHECK_NVJPEG(
      nvjpegDecodeBatchedInitialize(params.nvjpeg_handle,
                                    params.nvjpeg_state,
                                    batched_bitstreams.size(),
                                    1,
                                    params.fmt));

    CHECK_NVJPEG(nvjpegDecodeBatched(
      params.nvjpeg_handle,
      params.nvjpeg_state,
      batched_bitstreams.data(),
      batched_bitstreams_size.data(),
      batched_output.data(),
      params.stream));
  }

  if (!otherdecode_bitstreams.empty())
  {
    CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.nvjpeg_decoupled_state,
                                               params.device_buffer));
    int buffer_index = 0;
    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(params.nvjpeg_decode_params,
                                                   params.fmt));

    CHECK_NVJPEG(
      nvjpegJpegStreamParse(params.nvjpeg_handle,
                            otherdecode_bitstreams[0],
                            otherdecode_bitstreams_size[0],
                            0,
                            0,
                            params.jpeg_streams[buffer_index]));

    CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(params.nvjpeg_decoupled_state,
                                               params.pinned_buffers[buffer_index]));

    CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle,
                                      params.nvjpeg_decoder,
                                      params.nvjpeg_decoupled_state,
                                      params.nvjpeg_decode_params,
                                      params.jpeg_streams[buffer_index]));

    CHECK_CUDA(cudaStreamSynchronize(params.stream));

    CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle,
                                                  params.nvjpeg_decoder,
                                                  params.nvjpeg_decoupled_state,
                                                  params.jpeg_streams[buffer_index],
                                                  params.stream));

    buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

    CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle,
                                        params.nvjpeg_decoder,
                                        params.nvjpeg_decoupled_state,
                                        &otherdecode_output[0],
                                        params.stream));
  }
  CHECK_CUDA(cudaEventRecord(stopEvent, params.stream));

  CHECK_CUDA(cudaEventSynchronize(stopEvent));
  CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
  process_time = 0.001 * static_cast<double>(loopTime); // cudaEventElapsedTime returns milliseconds
  return true;
}

void release_buffer(nvjpegImage_t &ibuf)
{
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
    if (ibuf.channel[c])
    CHECK_CUDA(cudaFree(ibuf.channel[c]));

}

int main(int argc, const char *argv[])
{

  std::string filename = "/home/ne0/Pictures/test.jpg";
  StopWatchInterface *timer = nullptr;
  sdkCreateTimer(&timer);

  decode_params_t params {};
  params.fmt = NVJPEG_OUTPUT_RGB;

  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  nvjpegPinnedAllocator_t pinned_allocator = {&host_malloc, &host_free};
#ifdef CUDA11
  nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE,
                                         &dev_allocator,
                                         &pinned_allocator,
                                         NVJPEG_FLAGS_DEFAULT,
                                         &params.nvjpeg_handle);

  params.hw_decode_available = true;

  if( status == NVJPEG_STATUS_ARCH_MISMATCH) {
      std::cout<<"Hardware Decoder not supported. Falling back to default backend"<<std::endl;
#endif
  CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT,
                              &dev_allocator,
                              &pinned_allocator,
                              NVJPEG_FLAGS_DEFAULT,
                              &params.nvjpeg_handle));
  params.hw_decode_available = false;
#ifdef CUDA11
  } else {
      CHECK_NVJPEG(status);
  }
#endif
  CHECK_NVJPEG(
    nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));

  create_decoupled_api_handles(params);

  ///////////process

  // stream for decoding
  CHECK_CUDA(
    cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

  // output buffers
  nvjpegImage_t iout;
  // output buffer sizes, for convenience
  nvjpegImage_t isz;
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
  {
    iout.channel[c] = nullptr;
    iout.pitch[c] = 0;
    isz.pitch[c] = 0;
  }
  double test_time = 0;
  int warmup = 0;

  std::vector<char> raw_data;
  size_t raw_len;
  sdkStartTimer(&timer);
  read_image(filename, raw_data, raw_len);
  sdkStopTimer(&timer);
  auto cur_time = sdkGetAverageTimerValue(&timer) / 1000.0f;
  std::cout << "read_image:" << cur_time << "(s)" << std::endl;
  sdkResetTimer(&timer);

  size_t width, height;

  sdkStartTimer(&timer);
  prepare_buffers(raw_data, raw_len, params, width, height, iout, isz);
  sdkStopTimer(&timer);
  cur_time = sdkGetAverageTimerValue(&timer) / 1000.0f;
  std::cout << "prepare_buffers:" << cur_time << "(s)" << std::endl;
  double process_time;

  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  decode_image(raw_data, raw_len, iout, params, process_time);
  release_buffer(iout);
  sdkStopTimer(&timer);
  cur_time = sdkGetAverageTimerValue(&timer) / 1000.0f;
  std::cout << "decode_image:" << cur_time << "(s)" << std::endl;

  CHECK_CUDA(cudaStreamDestroy(params.stream));
  std::cout << "processing time:" << process_time << " (s)" << std::endl;
  //end process
  destroy_decoupled_api_handles(params);

  CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_state));
  CHECK_NVJPEG(nvjpegDestroy(params.nvjpeg_handle));

  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  cv::Mat mat = cv::imread(filename);
  sdkStopTimer(&timer);
  cur_time = sdkGetAverageTimerValue(&timer) / 1000.0f;
  std::cout << "imread_decode_jpeg:" << cur_time << "(s)" << std::endl;

  return EXIT_SUCCESS;
}