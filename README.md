# Neural Machine Translation using Quasi-RNN
Pytorch implementation of Neural Machine Translation using ["Quasi-Recurrent Neural Networks", ICLR 2017](https://arxiv.org/abs/1611.01576)

# Requirements
- NumPy >= 1.11.1
- Pytorch >= 0.2.0

# Usage Instructions
## **Codes**

- **layer.py** : Implementation of the quasi-recurrent layer
- **model.py**: Implementation of the Encoder-Decoder model using qrnn layer
- **train.py**: Code to train a NMT model
- **decode.py**: Code to translate a source file using a trained model

## **Training**
To train a quasi-rnn NMT model,
```ruby
$ python train.py --kernel_size 3 \
                  --hidden_size 640 \
                  --emb_size 500 \
                  --num_enc_symbols 30000 \
                  --num_dec_symbols 30000 ...
```

## **Decoding**
To run the trained model for translation,
```ruby
$ python eval.py  --model_path $path_to_model \
                  --decode_input $path_to_source \
                  --decode_output $path_to_output
                  --max_decode_step 300 \
                  --batch_size 30 ...                  
```
For simplicity, we used greedy decoding at each time step, not the beam search decoding.


# **Notes**

For more in-depth exploration, QRNN API for Pytorch is available: https://github.com/salesforce/pytorch-qrnn

For any comments and feedbacks, please email me at pjh0308@gmail.com or open an issue here.
