#include "turbojpeg.h"
#include <iostream>
#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>
#include <opencv2/opencv.hpp>

std::pair<cv::Mat, cv::Mat>
decompress_capture(k4a_capture_t capture, k4a_transformation_t transformation,
                   bool display = true) {
  cv::Mat color, depth;

  k4a_image_t k4a_color = k4a_capture_get_color_image(capture);
  k4a_image_t k4a_depth = k4a_capture_get_depth_image(capture);

  if (k4a_color == NULL || k4a_depth == NULL) {
    printf("skipping empty captures\n");
    return std::make_pair(color, depth);
  }
  k4a_image_format_t format = k4a_image_get_format(k4a_color);
  assert(format == K4A_IMAGE_FORMAT_COLOR_MJPG && "Image format is not MJPG");

  int color_width = k4a_image_get_width_pixels(k4a_color);
  int color_height = k4a_image_get_height_pixels(k4a_color);

  k4a_image_t k4a_uncompressed_color = NULL;
  if (K4A_RESULT_SUCCEEDED !=
      k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32, color_width, color_height,
                       color_width * 4 * (int)sizeof(uint8_t),
                       &k4a_uncompressed_color)) {
    printf("Failed to create uncompressed color image buffer\n");
    return std::make_pair(color, depth);
  }

  tjhandle tjHandle;
  tjHandle = tjInitDecompress();
  if (0 != tjDecompress2(
               tjHandle, k4a_image_get_buffer(k4a_color),
               static_cast<unsigned long>(k4a_image_get_size(k4a_color)),
               k4a_image_get_buffer(k4a_uncompressed_color), color_width,
               0, // pitch
               color_height, TJPF_BGRA, TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE)) {
    printf("Failed to decompress color image\n");
    return std::make_pair(color, depth);
  }
  tjDestroy(tjHandle);

  // transform color image into depth camera geometry
  k4a_image_t k4a_transformed_depth = NULL;
  if (K4A_RESULT_SUCCEEDED !=
      k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, color_width, color_height,
                       color_width * (int)sizeof(uint16_t),
                       &k4a_transformed_depth)) {
    printf("Failed to create transformed depth image buffer\n");
    return std::make_pair(color, depth);
  }

  if (K4A_RESULT_SUCCEEDED !=
      k4a_transformation_depth_image_to_color_camera(transformation, k4a_depth,
                                                     k4a_transformed_depth)) {
    printf("Failed to transform depth image\n");
    return std::make_pair(color, depth);
  }

  cv::Mat(color_height, color_width, CV_8UC4,
          k4a_image_get_buffer(k4a_uncompressed_color))
      .copyTo(color);
  cv::Mat(color_height, color_width, CV_16UC1,
          k4a_image_get_buffer(k4a_transformed_depth))
      .copyTo(depth);

  if (display) {
    cv::Mat cv_depth_intensity;
    depth.convertTo(cv_depth_intensity, CV_8UC1, 0.1);
    cv::imwrite("depth.png", depth);
    cv::imshow("color", color);
    cv::imshow("depth", cv_depth_intensity);
    cv::waitKey(30);
  }

  k4a_image_release(k4a_color);
  k4a_image_release(k4a_depth);
  k4a_image_release(k4a_uncompressed_color);
  k4a_image_release(k4a_transformed_depth);

  return std::make_pair(color, depth);
}

std::string k4a_read_tag(const k4a_playback_t &playback_handle,
                         const char *tag_name) {

  char res_buffer[256];
  size_t res_size = 256;

  k4a_buffer_result_t ret =
      k4a_playback_get_tag(playback_handle, tag_name, res_buffer, &res_size);
  if (ret == K4A_BUFFER_RESULT_SUCCEEDED) {
    printf("[%s] = %s\n", tag_name, res_buffer);
    return res_buffer;
  } else if (ret == K4A_BUFFER_RESULT_TOO_SMALL) {
    printf("[%s] tag content too long.\n", tag_name);
    return "";
  } else {
    printf("[%s] tag does not exist.\n", tag_name);
    return "";
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("please provide input [.mkv] filename.\n");
    return -1;
  }

  /** Open a playback **/
  k4a_playback_t playback_handle = NULL;
  if (k4a_playback_open(argv[1], &playback_handle) != K4A_RESULT_SUCCEEDED) {
    printf("Failed to open recording\n");
    return 1;
  }

  /** Print meta data **/
  uint64_t recording_length =
      k4a_playback_get_last_timestamp_usec(playback_handle);
  printf("Recording is %lld seconds long\n", recording_length / 1000000);

  k4a_read_tag(playback_handle, "K4A_DEVICE_SERIAL_NUMBER");
  k4a_read_tag(playback_handle, "K4A_DEPTH_MODE");
  k4a_read_tag(playback_handle, "K4A_COLOR_MODE");
  k4a_read_tag(playback_handle, "K4A_IR_MODE");
  k4a_read_tag(playback_handle, "K4A_IMU_MODE");
  k4a_read_tag(playback_handle, "K4A_CALIBRATION_FILE");

  /** Gain calibration **/
  k4a_calibration_t calibration;
  if (K4A_RESULT_SUCCEEDED !=
      k4a_playback_get_calibration(playback_handle, &calibration)) {
    printf("Failed to get calibration\n");
    return -1;
  }
  auto param = calibration.color_camera_calibration.intrinsics.parameters.param;
  printf("fx = %f, fy = %f, cx = %f, cy = %f\n", param.fx, param.fy, param.cx,
         param.cy);

  k4a_transformation_t transformation = k4a_transformation_create(&calibration);

  k4a_capture_t capture = NULL;
  k4a_stream_result_t result = K4A_STREAM_RESULT_SUCCEEDED;

  system("mkdir -p color");
  system("mkdir -p depth");
  int index = 0;
  char str_index[20];
  while (result == K4A_STREAM_RESULT_SUCCEEDED) {
    result = k4a_playback_get_next_capture(playback_handle, &capture);
    if (result == K4A_STREAM_RESULT_SUCCEEDED) {
      // Process capture here
      auto rgb_depth = decompress_capture(capture, transformation);

      if (!rgb_depth.first.empty() && !rgb_depth.second.empty()) {
        sprintf(str_index, "%05d.png", index);
        cv::imwrite("color/" + std::string(str_index), rgb_depth.first);
        cv::imwrite("depth/" + std::string(str_index), rgb_depth.second);
        ++index;
      }

      k4a_capture_release(capture);
    } else if (result == K4A_STREAM_RESULT_EOF) {
      // End of file reached
      break;
    }
  }
  if (result == K4A_STREAM_RESULT_FAILED) {
    printf("Failed to read entire recording\n");
    return 1;
  }

  k4a_playback_close(playback_handle);

  return 0;
}
