#ifndef LAYER_H
#define LAYER_H

typedef enum
{
    SAME = 0,
    VALID = 1,
} padding_t;

class CNN_layer
{
public:
    float ****out;      /** CNN输出 */
    float ****weights;
    float *bias;
    int batch,input_channel,output_channel,input_x,input_y,output_valid_x,output_valid_y;
    int ksize_x,ksize_y,strides_x,strides_y;
    int padding_l,padding_r,padding_up,padding_down;
    padding_t padding;
    void init(int input_batch, int input_m, int input_n, int input_c, int out_c, 
             float ****weights_init, float *bias_init, int ksize_m, int ksize_n,
             int strides_m,int strides_n, padding_t padding_mode);
    ~CNN_layer();
    void forward(float ****input);
};

#endif
/* end of file */