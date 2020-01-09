#include "layer.h"
#include "utils_other_copy.h"

/** 
 * \brief  卷积神经网络初始化函数
 *  
 * \param[in] input_batch      输入数据批次
 * \param[in] input_m          输入数据x方向尺寸
 * \param[in] input_n          输入数据y方向尺寸
 * \param[in] input_c          输入数据通道数
 * \param[in] out_c            输出数据通道数
 * \param[in] ****weights_init
 * \param[in] *bias_init
 * \param[in] ksize_m          x方向卷积核尺寸
 * \param[in] ksize_n          y方向卷积核尺寸
 * \param[in] strides_m        x方向卷积步长，默认1
 * \param[in] strides_n        y方向卷积步长，默认1
 * \param[in] padding_mode     默认1表示填充，0表示不填充
 */
void CNN_layer::init(int input_batch, int input_m, int input_n, int input_c, int out_c, 
             float ****weights_init, float *bias_init, int ksize_m = 3, int ksize_n = 3,
             int strides_m = 1,int strides_n = 1, padding_t padding_mode = SAME)
{
    /** \brief  以下均为给成员变量赋值*/
    batch          = input_batch;
    input_x        = input_m;
    input_y        = input_n;
    input_channel  = input_c;
    output_channel = out_c;

    ksize_x   = ksize_m;
    ksize_y   = ksize_n;
    strides_x = strides_m;
    strides_y = strides_n;
    padding   = padding_mode;
    weights   = weights_init;
    bias      = bias_init;
    output_valid_y = (input_y - ksize_y) / strides_y + 1;
    output_valid_x = (input_x - ksize_x) / strides_x + 1;
    /** \brief  上下左右四个边填充量计算，非偶数填充以右下优先 */
    if(padding == SAME) {
        padding_l = (ksize_x-1)/2;
        padding_r = (int)((float)(ksize_x-1)/2.0 + 0.5);
        padding_up = (ksize_y-1)/2;
        padding_down = (int)((float)(ksize_y-1)/2.0 + 0.5);
        out = get_mem<float>(batch,input_y,input_x,output_channel);
    }
    else {
        padding_l = 0;
        padding_r = 0;
        padding_up = 0;
        padding_down = 0;
        out = get_mem<float>(batch,output_valid_y,output_valid_x,output_channel);
    }
}

/** 
 * \brief  卷积神经网络析构函数
 * \details 释放分配空间
 * 
 */
CNN_layer::~CNN_layer()
{
    if(padding == SAME) {
        delete_mem<float>(out,batch,input_y,input_x,output_channel);
    }
    else {
        delete_mem<float>(out,batch,output_valid_y,output_valid_x,output_channel);
    }
}

/** 
 * \brief  卷积神经网络前向推理函数
 * \details 
 * 
 */
void CNN_layer::forward(float ****input)
{
    if(padding == SAME) {
        /** \brief  
         * 在SAME填充下，输出特征图与输入一样大小，
         * 故前四层嵌套for分别为batch、output_channel、input_y和input_x
         */
        for(int bt=0; bt<batch; bt++) {
            for (int c = 0; c<output_channel; c++) {
                for(int h=0; h<input_y; h++) {
                    for(int w=0; w<input_x; w++) {
                        /* 话不多说，先加上偏置再说 */
                        out[bt][h][w][c] = bias[c];
                        /* 输入通道循环，因为不同的“图像”输入通道对应不同的卷积核 */
                        for(int i=0; i<input_channel; i++) {
                            /* 以下两嵌套循环是卷积核内移动 */
                            for(int dir_y=0; dir_y<ksize_y; dir_y++) {
                                for(int dir_x=0; dir_x<ksize_x; dir_x++) {
                                    /** \brief  如果不满足if条件，说明该位置“像素”是填充0的，无需计算 */
                                    if((h+dir_y-padding_up)>=0 && (w+dir_x-padding_l)>=0 && (h+dir_y-padding_up)<input_y && (w+dir_x-padding_l)<input_x) {
                                        out[bt][h][w][c] += weights[dir_y][dir_x][i][c] * input[bt][h+dir_y-padding_up][w+dir_x-padding_l][i];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        /** \brief 注释参考以上 */ 
        for(int bt=0; bt<batch; bt++) {
            for(int c = 0; c<output_channel; c++) {
                for(int h=0; h<output_valid_y; h++) {
                    for(int w=0; w<output_valid_x; w++) {
                        out[bt][h][w][c] = bias[c];
                        for(int i=0; i<input_channel; i++) {
                            for(int dir_y=0; dir_y<ksize_y; dir_y++) {
                                for(int dir_x=0; dir_x<ksize_x; dir_x++) {
                                    out[bt][h][w][c] += weights[dir_y][dir_x][i][c] * input[bt][h+dir_y][w+dir_x][i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}