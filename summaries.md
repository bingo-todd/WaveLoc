·<!-- TOC -->

- [分带](#%E5%88%86%E5%B8%A6)
- [Gammatone 滤波器长度](#gammatone-%E6%BB%A4%E6%B3%A2%E5%99%A8%E9%95%BF%E5%BA%A6)

<!-- /TOC -->

## 分带
> a gammatone filter bank, which consists of 32 filters spanning between 70 and 7000 Hz
with peak gain set to 0 dB.

细节问题，文中给出的频率范围，70～7000Hz，是中心频率的范围、还是通带的频率范围？

<table>
<tr>
<td>CF_low=70</br>CF_high=7e3</td>
<td><img src='images/freq_low70_CF_high7e3.png'></td>
<tr>
<!-- <tr>
<td>freq_low=70</br>CF_high=7e3</td>
<td><img src='images/freq_low70_CF_high7e3.png'></td>
<tr>
<td>CF_low=70</br>freq_high=7e3</td>
<td><img src='images/cf_low70_freq_high7e3.png'></td>
<tr> -->
<td>freq_low=70</br>freq_high=7e3</td>
<td><img src='images/freq_low70_freq_high7e3.png'></td>
</tr>
</table>

**之前的实验，将滤波器的中心频率设置为[70,7000)**

## Gammatone 滤波器长度

  <img src='images/gammatone_kernel.png'>

```
  <table>
  <tr> <td>  </td > <td align=center> no padd </td> <td align=center> padd </td> </tr>
  <tr>
  <td>  </td>
  <td> <img src='pre_exps/images/kernel_test/gtf_kernel_no_padd/gtf_kernel_in_net.png'> </td>
  <td> <img src='pre_exps/images/kernel_test/gtf_kernel_padd/gtf_kernel_in_net.png'> </td>
  </tr>
```

  <tr>
  <td>  </td>
  <td> <img src='pre_exps/images/kernel_test/gtf_kernel_no_padd/layer1_norm_output.png'> </td>
  <td> <img src='pre_exps/images/kernel_test/gtf_kernel_padd/layer1_norm_output.png'> </td>
  </tr>
  </table>


  <table>
  <tr> <td>  </td > <td align=center> no aligned </td> <td align=center> aligned </td> </tr>
  <tr>
  <td>  </td>
  <td> <img src='pre_exps/images/kernel_test/gtf_kernel_no_padd_gtf/gtf_kernel_in_net.png'> </td>
  <td> <img src='pre_exps/images/kernel_test/gtf_kernel_no_padd_gtf_align/gtf_kernel_in_net.png'> </td>
  </tr>

  <tr>
  <td>  </td>
  <td> <img src='pre_exps/images/kernel_test/gtf_kernel_no_padd_gtf/layer1_norm_output.png'> </td>
  <td> <img src='pre_exps/images/kernel_test/gtf_kernel_no_padd_gtf_align/layer1_norm_output.png'> </td>
  </tr>
  </table>

  <!-- <center> <img src='pre_exps/images/kernel_test/gtf_kernel_no_padd_gtf/input.png'> </center> -->


## 结果
### 历史结果

<!-- <img src='images/k320_result.png'>
<img src='images/k320_320_result.png'>
<img src='images/k640_640_result.png'>
<img src='images/k960_960_result.png'> -->

<table>
<tr>
<td>
<img src='images/all_model_result.png'>
</td>
<td>
<img src='../baseline_v7/images/multi_test_errorbar.png'>
</td>
</tr>
</table>

Room B 的结果还是有点问题，重新跑一次实验


### 新跑的实验

  训练


  设置与论文一致

  <table>
    <tr>
      <td> train set </td> <td> <img src='images/basic_model_train_record_multi_run_train.png'></td>
    </tr>
    <tr>
     <td> Validation set </td>
     <td> <img src='images/basic_model_train_record_multi_run_valid.png'> </td>
    </tr>
    <tr>
      <td>Evaluation result</td><td>
        <img src='images/rmse_baic_multi_test.png'></td>
    </tr>
  </table>


  <!-- <table>
    <tr align=center>
        <td> run_0 </td>  <td> run_1 </td> <td> run_2 </td>
    </tr>
    <tr>
        <td> <img src='models/basic_0/rmse_multi_test.png'> </td>
        <td> <img src='models/basic/rmse_multi_test.png'> </td>
        <td> <img src='models/basic_2/rmse_multi_test.png'> </td>
    </tr>
  </table> -->



 Padding zeros to kernel
 <!-- <table>
 </table > -->
 <img src='models/padd/rmse_multi_test.png'>


<!-- <img src='/pre_exps/models/gtf_exp_run2/result/models_result.png'> -->
