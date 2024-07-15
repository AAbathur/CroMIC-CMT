# CroMIC-CMT
code for paper "Capturing Cross-Modal Semantics by Generating Comments
for Image-Text Contents"

## Install
```bash
pip install -r requirements.txt
pip install -e .
```

## Data Preparation

### Pre-training Data

1. Several samples are provided in "data/pre_training_data/example.txt".

2. The encoded textual posts and corresponding comments of these data samples are provided in "data/pre_training_data/example.pt". The conversion script is located in "data/pre_process_data".

3. The encoded visual post (image) are stored using the LMDB format in "example_image_lmdb".

### Fine-tuning Data
1. The datasets of downstream tasks:  FMIQA, VQA v2.0, AIC-ICC, Flickr30k-CN, MUGE(IC_Ecommerce) are available to researchers with simple applications.
Please download the datasets by yourself.

2. The text of the fine-tuning data is converted into .pt which is same as the format of the pre-training data.

We provided several examples of each dataset:

    FMIQA    --> data/downstream_data/FMIQA/FMIQA_example.pt
    VQA v2.0 --> data/downstream_data/VQA2.0/VQA2.0_example.pt
    AIC-ICC  --> data/downstream_data/AIC_ICC/AIC_ICC_example.pt

## Pre-training
The source code of baseline models [Unified VLP](https://github.com/LuoweiZhou/VLP) and [ViLT](https://github.com/dandelin/ViLT) are released officially. 
The modifications of Unified VLP and ViLT are provided in Experiment part of our paper.

Pre-training the new CroMIC-CMT model with below command:
```bash
# pre-training with single gpu
python pre_training.py --model_save_path <your/model/save/path> --text_data_folder <path/of/preprocessed/.pt/file> --image_data_folder <path/of/lmdb/file> --local_rank <device_id>

# pre-training with multiple gpu
python -m torch.distributed.launch --nproc_per_node <GPU NUM> pre_training.py --distributed --model_save_path <pre-trained/model/save/path> --text_data_folder <path/of/preprocessed/.pt/file> --image_data_folder <path/of/lmdb/file>

```

## Fine-tuning

```bash
python fine_tune.py --backbone_path <pre-trained/model/path> --task <down stream task> --device_id <device_id> --model_save_path <fine-tuned/model/save/path>

```

## Results on the downstream tasks

<table> 
    <tr>
    <td> Models </td>
    <td colspan="3"> MUGE </td>
    <td colspan="3"> AIC-ICC </td>
    <td colspan="3"> Flickr30k-CN </td>
    <td> FMIQA </td>
    <td> VQA v2.0 </td>
    <tr>
    <tr>
    <td> </td>
    <td> C </td> <td> B@4 </td> <td> R </td>
    <td> C </td> <td> B@4 </td> <td> R </td>
    <td> C </td> <td> B@4 </td> <td> R </td>
    <td> PassRate </td>
    <td> Acc </td>
    <tr>
    <td> Unified VLP </td> <td> 27.34</td> <td> 15.56</td> <td> 54.38</td> <td> 151.61</td> <td> 51.99</td> <td> 64.70</td> <td> 31.54</td> <td> 17.26</td> <td> 40.47</td> <td> 64.7%</td> <td> 51.9%</td>
    <tr>
    <td> ViLT </td> <td> 27.92</td> <td> 16.34</td> <td> 54.81</td> <td> 178.37</td> <td> 55.36</td> <td> 66.48</td> <td> 35.30</td> <td> 19.14</td> <td> 41.81</td> <td> 64.6%</td> <td> 54.1%</td>
    <tr>
    <td> Ours </td> <td> 34.20</td> <td> 18.66</td> <td> 55.92</td> <td> 186.20</td> <td> 57.45</td> <td> 68.57</td> <td> 39.59</td> <td> 22.62</td> <td> 43.57</td> <td> 69.2%</td> <td> 52.9%</td>
    <tr>

</table>


## License
This project is released under the [MIT License](LICENSE).