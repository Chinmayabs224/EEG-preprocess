
████████████████████████████████████████████████████████████
█  EEG SEIZURE DETECTION — INFERENCE / TESTING
█══════════════════════════════════════════════════════════█
█  Models dir  : models
█  Device      : cpu
█  Threshold   : 0.5
█  Models      : autoencoder, lstm, cnn, transformer
████████████████████████████████████████████████████████████

17:18:07 | INFO     | Loading trained models...
17:18:07 | INFO     |   ✓ Loaded autoencoder (2,309,280 params) from models/autoencoder_best.pt
17:18:07 | INFO     |   ✓ Loaded lstm (709,954 params) from models/lstm_best.pt
17:18:07 | INFO     |   ✓ Loaded cnn (274,690 params) from models/cnn_best.pt
/home/ubuntu/EEG-preprocess/training/model_transformer.py:114: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.transformer = nn.TransformerEncoder(
17:18:07 | INFO     |   ✓ Loaded transformer (547,266 params) from models/transformer_best.pt
17:18:07 | INFO     | Running inference on 686 file(s)...
  [001/686] chb01_01.npz                        🟢 Normal  ⚡ DETECTED
17:18:11 | INFO     | Saved timeline plot: test_plots/timeline_chb01_01.png
  [002/686] chb01_02.npz                        🟢 Normal  ⚡ DETECTED
17:18:14 | INFO     | Saved timeline plot: test_plots/timeline_chb01_02.png
  [003/686] chb01_03.npz                        🟢 Normal  ⚡ DETECTED
17:18:17 | INFO     | Saved timeline plot: test_plots/timeline_chb01_03.png
  [004/686] chb01_04.npz                        🟢 Normal  ⚡ DETECTED
17:18:21 | INFO     | Saved timeline plot: test_plots/timeline_chb01_04.png
  [005/686] chb01_05.npz                        🟢 Normal  ⚡ DETECTED
17:18:24 | INFO     | Saved timeline plot: test_plots/timeline_chb01_05.png
  [006/686] chb01_06.npz                        🟢 Normal  ⚡ DETECTED
17:18:27 | INFO     | Saved timeline plot: test_plots/timeline_chb01_06.png
  [007/686] chb01_07.npz                        🟢 Normal  ⚡ DETECTED
17:18:30 | INFO     | Saved timeline plot: test_plots/timeline_chb01_07.png
  [008/686] chb01_08.npz                        🟢 Normal  ⚡ DETECTED
17:18:34 | INFO     | Saved timeline plot: test_plots/timeline_chb01_08.png
  [009/686] chb01_09.npz                        🟢 Normal  ⚡ DETECTED
17:18:37 | INFO     | Saved timeline plot: test_plots/timeline_chb01_09.png
  [010/686] chb01_10.npz                        🟢 Normal  ⚡ DETECTED
17:18:40 | INFO     | Saved timeline plot: test_plots/timeline_chb01_10.png
  [011/686] chb01_11.npz                        🟢 Normal  ⚡ DETECTED
  [012/686] chb01_12.npz                        🟢 Normal  ⚡ DETECTED
  [013/686] chb01_13.npz                        🟢 Normal  ⚡ DETECTED
  [014/686] chb01_14.npz                        🟢 Normal  ⚡ DETECTED
  [015/686] chb01_15.npz                        🟢 Normal  ⚡ DETECTED
  [016/686] chb01_16.npz                        🟢 Normal  ⚡ DETECTED
  [017/686] chb01_17.npz                        🟢 Normal  ⚡ DETECTED
  [018/686] chb01_18.npz                        🟢 Normal  ⚡ DETECTED
  [019/686] chb01_19.npz                        🟢 Normal  ⚡ DETECTED
  [020/686] chb01_20.npz                        🟢 Normal  ⚡ DETECTED
  [021/686] chb01_21.npz                        🟢 Normal  ⚡ DETECTED
  [022/686] chb01_22.npz                        🟢 Normal  ⚡ DETECTED
  [023/686] chb01_23.npz                        🟢 Normal  ⚡ DETECTED
  [024/686] chb01_24.npz                        🟢 Normal  ⚡ DETECTED
  [025/686] chb01_25.npz                        🟢 Normal  ⚡ DETECTED
  [026/686] chb01_26.npz                        🟢 Normal  ⚡ DETECTED
  [027/686] chb01_27.npz                        🟢 Normal  ⚡ DETECTED
  [028/686] chb01_29.npz                        🟢 Normal  ⚡ DETECTED
  [029/686] chb01_30.npz                        🟢 Normal  ⚡ DETECTED
  [030/686] chb01_31.npz                        🟢 Normal  ⚡ DETECTED
  [031/686] chb01_32.npz                        🟢 Normal  ⚡ DETECTED
  [032/686] chb01_33.npz                        🟢 Normal  ⚡ DETECTED
  [033/686] chb01_34.npz                        🟢 Normal  ⚡ DETECTED
  [034/686] chb01_36.npz                        🟢 Normal  ⚡ DETECTED
  [035/686] chb01_37.npz                        🟢 Normal  ⚡ DETECTED
  [036/686] chb01_38.npz                        🟢 Normal  ⚡ DETECTED
  [037/686] chb01_39.npz                        🟢 Normal  ⚡ DETECTED
  [038/686] chb01_40.npz                        🟢 Normal  ⚡ DETECTED
  [039/686] chb01_41.npz                        🟢 Normal  ⚡ DETECTED
  [040/686] chb01_42.npz                        🟢 Normal  ⚡ DETECTED
  [041/686] chb01_43.npz                        🟢 Normal  ⚡ DETECTED
  [042/686] chb01_46.npz                        🟢 Normal  ⚡ DETECTED
  [043/686] chb02_01.npz                        🟢 Normal  ⚡ DETECTED
  [044/686] chb02_02.npz                        🟢 Normal  ⚡ DETECTED
  [045/686] chb02_03.npz                        🟢 Normal  ⚡ DETECTED
  [046/686] chb02_04.npz                        🟢 Normal  ⚡ DETECTED
  [047/686] chb02_05.npz                        🟢 Normal  ⚡ DETECTED
  [048/686] chb02_06.npz                        🟢 Normal  ⚡ DETECTED
  [049/686] chb02_07.npz                        🟢 Normal  ⚡ DETECTED
  [050/686] chb02_08.npz                        🟢 Normal  ⚡ DETECTED
  [051/686] chb02_09.npz                        🟢 Normal  ⚡ DETECTED
  [052/686] chb02_10.npz                        🟢 Normal  ⚡ DETECTED
  [053/686] chb02_11.npz                        🟢 Normal  ⚡ DETECTED
  [054/686] chb02_12.npz                        🟢 Normal  ⚡ DETECTED
  [055/686] chb02_13.npz                        🟢 Normal  ⚡ DETECTED
  [056/686] chb02_14.npz                        🟢 Normal  ⚡ DETECTED
  [057/686] chb02_15.npz                        🟢 Normal  ⚡ DETECTED
  [058/686] chb02_16+.npz                       🟢 Normal  ⚡ DETECTED
  [059/686] chb02_16.npz                        🟢 Normal  ⚡ DETECTED
  [060/686] chb02_17.npz                        🟢 Normal  ⚡ DETECTED
  [061/686] chb02_18.npz                        🟢 Normal  ⚡ DETECTED
  [062/686] chb02_19.npz                        🟢 Normal  ⚡ DETECTED
  [063/686] chb02_20.npz                        🟢 Normal  ⚡ DETECTED
  [064/686] chb02_21.npz                        🟢 Normal  ⚡ DETECTED
  [065/686] chb02_22.npz                        🟢 Normal  ⚡ DETECTED
  [066/686] chb02_23.npz                        🟢 Normal  ⚡ DETECTED
  [067/686] chb02_24.npz                        🟢 Normal  ⚡ DETECTED
  [068/686] chb02_25.npz                        🟢 Normal  ⚡ DETECTED
  [069/686] chb02_26.npz                        🟢 Normal  ⚡ DETECTED
  [070/686] chb02_27.npz                        🟢 Normal  ⚡ DETECTED
  [071/686] chb02_28.npz                        🟢 Normal  ⚡ DETECTED
  [072/686] chb02_29.npz                        🟢 Normal  ⚡ DETECTED
  [073/686] chb02_30.npz                        🟢 Normal  ⚡ DETECTED
  [074/686] chb02_31.npz                        🟢 Normal  ⚡ DETECTED
  [075/686] chb02_32.npz                        🟢 Normal  ⚡ DETECTED
  [076/686] chb02_33.npz                        🟢 Normal  ⚡ DETECTED
  [077/686] chb02_34.npz                        🟢 Normal  ⚡ DETECTED
  [078/686] chb02_35.npz                        🟢 Normal  ⚡ DETECTED
  [079/686] chb03_01.npz                        🟢 Normal  ⚡ DETECTED
  [080/686] chb03_02.npz                        🟢 Normal  ⚡ DETECTED
  [081/686] chb03_03.npz                        🟢 Normal  ⚡ DETECTED
  [082/686] chb03_04.npz                        🟢 Normal  ⚡ DETECTED
  [083/686] chb03_05.npz                        🟢 Normal  ⚡ DETECTED
  [084/686] chb03_06.npz                        🟢 Normal  ⚡ DETECTED
  [085/686] chb03_07.npz                        🟢 Normal  ⚡ DETECTED
  [086/686] chb03_08.npz                        🟢 Normal  ⚡ DETECTED
  [087/686] chb03_09.npz                        🟢 Normal  ⚡ DETECTED
  [088/686] chb03_10.npz                        🟢 Normal  ⚡ DETECTED
  [089/686] chb03_11.npz                        🟢 Normal  ⚡ DETECTED
  [090/686] chb03_12.npz                        🟢 Normal  ⚡ DETECTED
  [091/686] chb03_13.npz                        🟢 Normal  ⚡ DETECTED
  [092/686] chb03_14.npz                        🟢 Normal  ⚡ DETECTED
  [093/686] chb03_15.npz                        🟢 Normal  ⚡ DETECTED
  [094/686] chb03_16.npz                        🟢 Normal  ⚡ DETECTED
  [095/686] chb03_17.npz                        🟢 Normal  ⚡ DETECTED
  [096/686] chb03_18.npz                        🟢 Normal  ⚡ DETECTED
  [097/686] chb03_19.npz                        🟢 Normal  ⚡ DETECTED
  [098/686] chb03_20.npz                        🟢 Normal  ⚡ DETECTED
  [099/686] chb03_21.npz                        🟢 Normal  ⚡ DETECTED
  [100/686] chb03_22.npz                        🟢 Normal  ⚡ DETECTED
  [101/686] chb03_23.npz                        🟢 Normal  ⚡ DETECTED
  [102/686] chb03_24.npz                        🟢 Normal  ⚡ DETECTED
  [103/686] chb03_25.npz                        🟢 Normal  ⚡ DETECTED
  [104/686] chb03_26.npz                        🟢 Normal  ⚡ DETECTED
  [105/686] chb03_27.npz                        🟢 Normal  ⚡ DETECTED
^CTraceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/ubuntu/EEG-preprocess/training/predict.py", line 538, in <module>
    main()
  File "/home/ubuntu/EEG-preprocess/training/predict.py", line 433, in main
    result = predict_single_file(
  File "/home/ubuntu/EEG-preprocess/training/predict.py", line 198, in predict_single_file
    logits = model(batch)
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/EEG-preprocess/training/model_transformer.py", line 150, in forward
    x = self.transformer(x)  # (B, T+1, d_model)
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/transformer.py", line 540, in forward
    output = mod(
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/transformer.py", line 921, in forward
    return torch._transformer_encoder_layer_fwd(
KeyboardInterrupt

ubuntu@ip-172-31-10-237:~/EEG-preprocess$ cd models/
ubuntu@ip-172-31-10-237:~/EEG-preprocess/models$ ls
all_results.json     autoencoder_metrics.json  cnn_metrics.json  lstm_metrics.json    transformer_metrics.json
autoencoder_best.pt  cnn_best.pt               lstm_best.pt      transformer_best.pt
ubuntu@ip-172-31-10-237:~/EEG-preprocess/models$ code .
Command 'code' not found, but can be installed with:
sudo snap install code
ubuntu@ip-172-31-10-237:~/EEG-preprocess/models$ ^C
ubuntu@ip-172-31-10-237:~/EEG-preprocess/models$ grep -h "accuracy\|auc_roc\|f1" *_metrics.json
  "accuracy": 0.9471298742552549,
  "f1": 0.007851119511485897,
  "auc_roc": 0.5417904826700709,
  "accuracy": 0.7183184788376696,
  "f1": 0.018914251794290647,
  "auc_roc": 0.8597332433636926
  "accuracy": 0.949585886837477,
  "f1": 0.06816554489474438,
  "auc_roc": 0.830713414384191
  "accuracy": 0.8679951344608782,
  "f1": 0.03505691793622926,
  "auc_roc": 0.879110909170641
ubuntu@ip-172-31-10-237:~/EEG-preprocess/models$ cat all_results.json
{
  "autoencoder": {
    "accuracy": 0.9471298742552549,
    "precision": 0.004183452122714595,
    "recall": 0.06367924528301887,
    "f1": 0.007851119511485897,
    "specificity": 0.9500415866673921,
    "true_positives": 54,
    "false_positives": 12854,
    "true_negatives": 244440,
    "false_negatives": 794,
    "auc_roc": 0.5417904826700709,
    "threshold": 0.000426757731474936,
    "mean_error_normal": 0.00032860395731404424,
    "mean_error_seizure": 0.0003365762531757355
  },
  "lstm": {
    "accuracy": 0.949585886837477,
    "precision": 0.03628601921024546,
    "recall": 0.5613207547169812,
    "f1": 0.06816554489474438,
    "specificity": 0.9508655468063771,
    "true_positives": 476,
    "false_positives": 12642,
    "true_negatives": 244652,
    "false_negatives": 372,
    "auc_roc": 0.830713414384191
  },
  "cnn": {
    "accuracy": 0.7183184788376696,
    "precision": 0.009566570227632513,
    "recall": 0.8266509433962265,
    "f1": 0.018914251794290647,
    "specificity": 0.7179614725461774,
    "true_positives": 701,
    "false_positives": 72575,
    "true_negatives": 184748,
    "false_negatives": 147,
    "auc_roc": 0.8597332433636926
  },
  "transformer": {
    "accuracy": 0.8679951344608782,
    "precision": 0.017959728428016016,
    "recall": 0.7299528301886793,
    "f1": 0.03505691793622926,
    "specificity": 0.8684500998857339,
    "true_positives": 619,
    "false_positives": 33847,
    "true_negatives": 223447,
    "false_negatives": 229,
    "auc_roc": 0.879110909170641
  },
  "ensemble": {
    "accuracy": 0.9775278722563551,
    "precision": 0.061293179805137286,
    "recall": 0.4080188679245283,
    "f1": 0.10657631295241028,
    "specificity": 0.9794048831298048,
    "true_positives": 346,
    "false_positives": 5299,
    "true_negatives": 251995,
    "false_negatives": 502,
    "auc_roc": 0.8606184590464092
  },
  "ensemble_weights": {
    "lstm": 0.30000000000000004,
    "transformer": 0.1,
    "cnn": 0.1,
    "autoencoder": 0.5
  }
}ubuntu@ip-172-31-10-237:~/EEG-preprocess/models$ cd ..
ubuntu@ip-172-31-10-237:~/EEG-preprocess$ python3 -m training.predict --spikes_dir output/spikes --models_dir models --plots_dir test_plots

████████████████████████████████████████████████████████████
█  EEG SEIZURE DETECTION — INFERENCE / TESTING
█══════════════════════════════════════════════════════════█
█  Models dir  : models
█  Device      : cpu
█  Threshold   : 0.5
█  Models      : autoencoder, lstm, cnn, transformer
████████████████████████████████████████████████████████████

17:26:11 | INFO     | Loading trained models...
17:26:11 | INFO     |   ✓ Loaded autoencoder (2,309,280 params) from models/autoencoder_best.pt
17:26:11 | INFO     |   ✓ Loaded lstm (709,954 params) from models/lstm_best.pt
17:26:11 | INFO     |   ✓ Loaded cnn (274,690 params) from models/cnn_best.pt
/home/ubuntu/EEG-preprocess/training/model_transformer.py:114: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.transformer = nn.TransformerEncoder(
17:26:11 | INFO     |   ✓ Loaded transformer (547,266 params) from models/transformer_best.pt
17:26:11 | INFO     | Running inference on 686 file(s)...
  [001/686] chb01_01.npz                        🟢 Normal  ⚡ DETECTED
17:26:15 | INFO     | Saved timeline plot: test_plots/timeline_chb01_01.png
  [002/686] chb01_02.npz                        🟢 Normal  ⚡ DETECTED
17:26:18 | INFO     | Saved timeline plot: test_plots/timeline_chb01_02.png
  [003/686] chb01_03.npz                        🟢 Normal  ⚡ DETECTED
17:26:22 | INFO     | Saved timeline plot: test_plots/timeline_chb01_03.png
  [004/686] chb01_04.npz                        🟢 Normal  ⚡ DETECTED
17:26:26 | INFO     | Saved timeline plot: test_plots/timeline_chb01_04.png
  [005/686] chb01_05.npz                        🟢 Normal  ⚡ DETECTED
17:26:29 | INFO     | Saved timeline plot: test_plots/timeline_chb01_05.png
  [006/686] chb01_06.npz                        🟢 Normal  ⚡ DETECTED
17:26:32 | INFO     | Saved timeline plot: test_plots/timeline_chb01_06.png
  [007/686] chb01_07.npz                        🟢 Normal  ⚡ DETECTED
17:26:36 | INFO     | Saved timeline plot: test_plots/timeline_chb01_07.png
  [008/686] chb01_08.npz                        🟢 Normal  ⚡ DETECTED
17:26:40 | INFO     | Saved timeline plot: test_plots/timeline_chb01_08.png
  [009/686] chb01_09.npz                        🟢 Normal  ⚡ DETECTED
17:26:43 | INFO     | Saved timeline plot: test_plots/timeline_chb01_09.png
  [010/686] chb01_10.npz                        🟢 Normal  ⚡ DETECTED
17:26:47 | INFO     | Saved timeline plot: test_plots/timeline_chb01_10.png
  [011/686] chb01_11.npz                        🟢 Normal  ⚡ DETECTED
  [012/686] chb01_12.npz                        🟢 Normal  ⚡ DETECTED
  [013/686] chb01_13.npz                        🟢 Normal  ⚡ DETECTED
  [014/686] chb01_14.npz                        🟢 Normal  ⚡ DETECTED
  [015/686] chb01_15.npz                        🟢 Normal  ⚡ DETECTED
  [016/686] chb01_16.npz                        🟢 Normal  ⚡ DETECTED
  [017/686] chb01_17.npz                        🟢 Normal  ⚡ DETECTED
  [018/686] chb01_18.npz                        🟢 Normal  ⚡ DETECTED
  [019/686] chb01_19.npz                        🟢 Normal  ⚡ DETECTED
  [020/686] chb01_20.npz                        🟢 Normal  ⚡ DETECTED
  [021/686] chb01_21.npz                        🟢 Normal  ⚡ DETECTED
  [022/686] chb01_22.npz                        🟢 Normal  ⚡ DETECTED
  [023/686] chb01_23.npz                        🟢 Normal  ⚡ DETECTED
  [024/686] chb01_24.npz                        🟢 Normal  ⚡ DETECTED
  [025/686] chb01_25.npz                        🟢 Normal  ⚡ DETECTED
  [026/686] chb01_26.npz                        🟢 Normal  ⚡ DETECTED
  [027/686] chb01_27.npz                        🟢 Normal  ⚡ DETECTED
  [028/686] chb01_29.npz                        🟢 Normal  ⚡ DETECTED
  [029/686] chb01_30.npz                        🟢 Normal  ⚡ DETECTED
  [030/686] chb01_31.npz                        🟢 Normal  ⚡ DETECTED
  [031/686] chb01_32.npz                        🟢 Normal  ⚡ DETECTED
  [032/686] chb01_33.npz                        🟢 Normal  ⚡ DETECTED
  [033/686] chb01_34.npz                        🟢 Normal  ⚡ DETECTED
  [034/686] chb01_36.npz                        🟢 Normal  ⚡ DETECTED
  [035/686] chb01_37.npz                        🟢 Normal  ⚡ DETECTED
  [036/686] chb01_38.npz                        🟢 Normal  ⚡ DETECTED
  [037/686] chb01_39.npz                        🟢 Normal  ⚡ DETECTED
  [038/686] chb01_40.npz                        🟢 Normal  ⚡ DETECTED
  [039/686] chb01_41.npz                        🟢 Normal  ⚡ DETECTED
  [040/686] chb01_42.npz                        🟢 Normal  ⚡ DETECTED
  [041/686] chb01_43.npz                        🟢 Normal  ⚡ DETECTED
  [042/686] chb01_46.npz                        🟢 Normal  ⚡ DETECTED
  [043/686] chb02_01.npz                        🟢 Normal  ⚡ DETECTED
  [044/686] chb02_02.npz                        🟢 Normal  ⚡ DETECTED
  [045/686] chb02_03.npz                        🟢 Normal  ⚡ DETECTED
  [046/686] chb02_04.npz                        🟢 Normal  ⚡ DETECTED
  [047/686] chb02_05.npz                        🟢 Normal  ⚡ DETECTED
  [048/686] chb02_06.npz                        🟢 Normal  ⚡ DETECTED
  [049/686] chb02_07.npz                        🟢 Normal  ⚡ DETECTED
  [050/686] chb02_08.npz                        🟢 Normal  ⚡ DETECTED
  [051/686] chb02_09.npz                        🟢 Normal  ⚡ DETECTED
  [052/686] chb02_10.npz                        🟢 Normal  ⚡ DETECTED
  [053/686] chb02_11.npz                        🟢 Normal  ⚡ DETECTED
  [054/686] chb02_12.npz                        🟢 Normal  ⚡ DETECTED
  [055/686] chb02_13.npz                        🟢 Normal  ⚡ DETECTED
  [056/686] chb02_14.npz                        🟢 Normal  ⚡ DETECTED
  [057/686] chb02_15.npz                        🟢 Normal  ⚡ DETECTED
  [058/686] chb02_16+.npz                       🟢 Normal  ⚡ DETECTED
  [059/686] chb02_16.npz                        🟢 Normal  ⚡ DETECTED
  [060/686] chb02_17.npz                        🟢 Normal  ⚡ DETECTED
  [061/686] chb02_18.npz                        🟢 Normal  ⚡ DETECTED
  [062/686] chb02_19.npz                        🟢 Normal  ⚡ DETECTED
  [063/686] chb02_20.npz                        🟢 Normal  ⚡ DETECTED
  [064/686] chb02_21.npz                        🟢 Normal  ⚡ DETECTED
  [065/686] chb02_22.npz                        🟢 Normal  ⚡ DETECTED
  [066/686] chb02_23.npz                        🟢 Normal  ⚡ DETECTED
  [067/686] chb02_24.npz                        🟢 Normal  ⚡ DETECTED
  [068/686] chb02_25.npz                        🟢 Normal  ⚡ DETECTED
  [069/686] chb02_26.npz                        🟢 Normal  ⚡ DETECTED
  [070/686] chb02_27.npz                        🟢 Normal  ⚡ DETECTED
  [071/686] chb02_28.npz                        🟢 Normal  ⚡ DETECTED
  [072/686] chb02_29.npz                        🟢 Normal  ⚡ DETECTED
  [073/686] chb02_30.npz                        🟢 Normal  ⚡ DETECTED
  [074/686] chb02_31.npz                        🟢 Normal  ⚡ DETECTED
  [075/686] chb02_32.npz                        🟢 Normal  ⚡ DETECTED
  [076/686] chb02_33.npz                        🟢 Normal  ⚡ DETECTED
  [077/686] chb02_34.npz                        🟢 Normal  ⚡ DETECTED
  [078/686] chb02_35.npz                        🟢 Normal  ⚡ DETECTED
  [079/686] chb03_01.npz                        🟢 Normal  ⚡ DETECTED
  [080/686] chb03_02.npz                        🟢 Normal  ⚡ DETECTED
  [081/686] chb03_03.npz                        🟢 Normal  ⚡ DETECTED
  [082/686] chb03_04.npz                        🟢 Normal  ⚡ DETECTED
  [083/686] chb03_05.npz                        🟢 Normal  ⚡ DETECTED
  [084/686] chb03_06.npz                        🟢 Normal  ⚡ DETECTED
  [085/686] chb03_07.npz                        🟢 Normal  ⚡ DETECTED
  [086/686] chb03_08.npz                        🟢 Normal  ⚡ DETECTED
  [087/686] chb03_09.npz                        🟢 Normal  ⚡ DETECTED
  [088/686] chb03_10.npz                        🟢 Normal  ⚡ DETECTED
  [089/686] chb03_11.npz                        🟢 Normal  ⚡ DETECTED
  [090/686] chb03_12.npz                        🟢 Normal  ⚡ DETECTED
  [091/686] chb03_13.npz                        🟢 Normal  ⚡ DETECTED
  [092/686] chb03_14.npz                        🟢 Normal  ⚡ DETECTED
  [093/686] chb03_15.npz                        🟢 Normal  ⚡ DETECTED
  [094/686] chb03_16.npz                        🟢 Normal  ⚡ DETECTED
  [095/686] chb03_17.npz                        🟢 Normal  ⚡ DETECTED
  [096/686] chb03_18.npz                        🟢 Normal  ⚡ DETECTED
  [097/686] chb03_19.npz                        🟢 Normal  ⚡ DETECTED
  [098/686] chb03_20.npz                        🟢 Normal  ⚡ DETECTED
  [099/686] chb03_21.npz                        🟢 Normal  ⚡ DETECTED
  [100/686] chb03_22.npz                        🟢 Normal  ⚡ DETECTED
  [101/686] chb03_23.npz                        🟢 Normal  ⚡ DETECTED
  [102/686] chb03_24.npz                        🟢 Normal  ⚡ DETECTED
  [103/686] chb03_25.npz                        🟢 Normal  ⚡ DETECTED
  [104/686] chb03_26.npz                        🟢 Normal  ⚡ DETECTED
  [105/686] chb03_27.npz                        🟢 Normal  ⚡ DETECTED
  [106/686] chb03_28.npz                        🟢 Normal  ⚡ DETECTED
  [107/686] chb03_29.npz                        🟢 Normal  ⚡ DETECTED
  [108/686] chb03_30.npz                        🟢 Normal  ⚡ DETECTED
  [109/686] chb03_31.npz                        🟢 Normal  ⚡ DETECTED
  [110/686] chb03_32.npz                        🟢 Normal  ⚡ DETECTED
  [111/686] chb03_33.npz                        🟢 Normal  ⚡ DETECTED
  [112/686] chb03_34.npz                        🟢 Normal  ⚡ DETECTED
  [113/686] chb03_35.npz                        🟢 Normal  ⚡ DETECTED
  [114/686] chb03_36.npz                        🟢 Normal  ⚡ DETECTED
  [115/686] chb03_37.npz                        🟢 Normal  ⚡ DETECTED
  [116/686] chb03_38.npz                        🟢 Normal  ⚡ DETECTED
  [117/686] chb04_01.npz                        🟢 Normal  ⚡ DETECTED
  [118/686] chb04_02.npz                        🟢 Normal  ⚡ DETECTED
  [119/686] chb04_03.npz                        🟢 Normal  ⚡ DETECTED
  [120/686] chb04_04.npz                        🟢 Normal  ⚡ DETECTED
  [121/686] chb04_05.npz                        🟢 Normal  ⚡ DETECTED
  [122/686] chb04_06.npz                        🟢 Normal  ⚡ DETECTED
  [123/686] chb04_07.npz                        🟢 Normal  ⚡ DETECTED
  [124/686] chb04_08.npz                        🟢 Normal  ⚡ DETECTED
  [125/686] chb04_09.npz                        🟢 Normal  ⚡ DETECTED
  [126/686] chb04_10.npz                        🟢 Normal  ⚡ DETECTED
  [127/686] chb04_11.npz                        🟢 Normal  ⚡ DETECTED
  [128/686] chb04_12.npz                        🟢 Normal  ⚡ DETECTED
  [129/686] chb04_13.npz                        🟢 Normal  ⚡ DETECTED
  [130/686] chb04_14.npz                        🟢 Normal  ⚡ DETECTED
  [131/686] chb04_15.npz                        🟢 Normal  ⚡ DETECTED
  [132/686] chb04_16.npz                        🟢 Normal  ⚡ DETECTED
  [133/686] chb04_17.npz                        🟢 Normal  ⚡ DETECTED
  [134/686] chb04_18.npz                        🟢 Normal  ⚡ DETECTED
  [135/686] chb04_19.npz                        🟢 Normal  ⚡ DETECTED
  [136/686] chb04_21.npz                        🟢 Normal  ⚡ DETECTED
  [137/686] chb04_22.npz                        🟢 Normal  ⚡ DETECTED
  [138/686] chb04_23.npz                        🟢 Normal  ⚡ DETECTED
  [139/686] chb04_24.npz                        🟢 Normal  ⚡ DETECTED
  [140/686] chb04_25.npz                        🟢 Normal  ⚡ DETECTED
  [141/686] chb04_26.npz                        🟢 Normal  ⚡ DETECTED
  [142/686] chb04_27.npz                        🟢 Normal  ⚡ DETECTED
  [143/686] chb04_28.npz                        🟢 Normal  ⚡ DETECTED
  [144/686] chb04_29.npz                        🟢 Normal  ⚡ DETECTED
  [145/686] chb04_30.npz                        🟢 Normal  ⚡ DETECTED
  [146/686] chb04_31.npz                        🟢 Normal  ⚡ DETECTED
  [147/686] chb04_32.npz                        🟢 Normal  ⚡ DETECTED
  [148/686] chb04_33.npz                        🟢 Normal  ⚡ DETECTED
  [149/686] chb04_34.npz                        🟢 Normal  ⚡ DETECTED
  [150/686] chb04_35.npz                        🟢 Normal  ⚡ DETECTED
  [151/686] chb04_36.npz                        🟢 Normal  ⚡ DETECTED
  [152/686] chb04_37.npz                        🟢 Normal  ⚡ DETECTED
  [153/686] chb04_38.npz                        🟢 Normal  ⚡ DETECTED
  [154/686] chb04_39.npz                        🟢 Normal  ⚡ DETECTED
  [155/686] chb04_40.npz                        🟢 Normal  ⚡ DETECTED
  [156/686] chb04_41.npz                        🟢 Normal  ⚡ DETECTED
  [157/686] chb04_42.npz                        🟢 Normal  ⚡ DETECTED
  [158/686] chb04_43.npz                        🟢 Normal  ⚡ DETECTED
  [159/686] chb05_01.npz                        🟢 Normal  ⚡ DETECTED
  [160/686] chb05_02.npz                        🟢 Normal  ⚡ DETECTED
  [161/686] chb05_03.npz                        🟢 Normal  ⚡ DETECTED
  [162/686] chb05_04.npz                        🟢 Normal  ⚡ DETECTED
  [163/686] chb05_05.npz                        🟢 Normal  ⚡ DETECTED
  [164/686] chb05_06.npz                        🟢 Normal  ⚡ DETECTED
  [165/686] chb05_07.npz                        🟢 Normal  ⚡ DETECTED
  [166/686] chb05_08.npz                        🟢 Normal  ⚡ DETECTED
  [167/686] chb05_09.npz                        🟢 Normal  ⚡ DETECTED
  [168/686] chb05_10.npz                        🟢 Normal  ⚡ DETECTED
  [169/686] chb05_11.npz                        🟢 Normal  ⚡ DETECTED
  [170/686] chb05_12.npz                        🟢 Normal  ⚡ DETECTED
  [171/686] chb05_13.npz                        🟢 Normal  ⚡ DETECTED
  [172/686] chb05_14.npz                        🟢 Normal  ⚡ DETECTED
  [173/686] chb05_15.npz                        🟢 Normal  ⚡ DETECTED
  [174/686] chb05_16.npz                        🟢 Normal  ⚡ DETECTED
  [175/686] chb05_17.npz                        🟢 Normal  ⚡ DETECTED
  [176/686] chb05_18.npz                        🟢 Normal  ⚡ DETECTED
  [177/686] chb05_19.npz                        🟢 Normal  ⚡ DETECTED
  [178/686] chb05_20.npz                        🟢 Normal  ⚡ DETECTED
  [179/686] chb05_21.npz                        🟢 Normal  ⚡ DETECTED
  [180/686] chb05_22.npz                        🟢 Normal  ⚡ DETECTED
  [181/686] chb05_23.npz                        🟢 Normal  ⚡ DETECTED
  [182/686] chb05_24.npz                        🟢 Normal  ⚡ DETECTED
  [183/686] chb05_25.npz                        🟢 Normal  ⚡ DETECTED
  [184/686] chb05_26.npz                        🟢 Normal  ⚡ DETECTED
  [185/686] chb05_27.npz                        🟢 Normal  ⚡ DETECTED
  [186/686] chb05_28.npz                        🟢 Normal  ⚡ DETECTED
  [187/686] chb05_29.npz                        🟢 Normal  ⚡ DETECTED
  [188/686] chb05_30.npz                        🟢 Normal  ⚡ DETECTED
  [189/686] chb05_31.npz                        🟢 Normal  ⚡ DETECTED
  [190/686] chb05_32.npz                        🟢 Normal  ⚡ DETECTED
  [191/686] chb05_33.npz                        🟢 Normal  ⚡ DETECTED
  [192/686] chb05_34.npz                        🟢 Normal  ⚡ DETECTED
  [193/686] chb05_35.npz                        🟢 Normal  ⚡ DETECTED
  [194/686] chb05_36.npz                        🟢 Normal  ⚡ DETECTED
  [195/686] chb05_37.npz                        🟢 Normal  ⚡ DETECTED
  [196/686] chb05_38.npz                        🟢 Normal  ⚡ DETECTED
  [197/686] chb05_39.npz                        🟢 Normal  ⚡ DETECTED
  [198/686] chb06_01.npz                        🟢 Normal  ⚡ DETECTED
  [199/686] chb06_02.npz                        🟢 Normal  ⚡ DETECTED
  [200/686] chb06_03.npz                        🟢 Normal  ⚡ DETECTED
  [201/686] chb06_04.npz                        🟢 Normal  ⚡ DETECTED
  [202/686] chb06_05.npz                        🟢 Normal  ⚡ DETECTED
  [203/686] chb06_06.npz                        🟢 Normal  ⚡ DETECTED
  [204/686] chb06_07.npz                        🟢 Normal  ⚡ DETECTED
  [205/686] chb06_08.npz                        🟢 Normal  ⚡ DETECTED
  [206/686] chb06_09.npz                        🟢 Normal  ⚡ DETECTED
  [207/686] chb06_10.npz                        🟢 Normal  ⚡ DETECTED
  [208/686] chb06_12.npz                        🟢 Normal  ⚡ DETECTED
  [209/686] chb06_13.npz                        🟢 Normal  ⚡ DETECTED
  [210/686] chb06_14.npz                        🟢 Normal  ⚡ DETECTED
  [211/686] chb06_15.npz                        🟢 Normal  ⚡ DETECTED
  [212/686] chb06_16.npz                        🟢 Normal  ⚡ DETECTED
  [213/686] chb06_17.npz                        🟢 Normal  ⚡ DETECTED
  [214/686] chb06_18.npz                        🟢 Normal  ⚡ DETECTED
  [215/686] chb06_24.npz                        🟢 Normal  ⚡ DETECTED
  [216/686] chb07_01.npz                        🟢 Normal  ⚡ DETECTED
  [217/686] chb07_02.npz                        🟢 Normal  ⚡ DETECTED
  [218/686] chb07_03.npz                        🟢 Normal  ⚡ DETECTED
  [219/686] chb07_04.npz                        🟢 Normal  ⚡ DETECTED
  [220/686] chb07_05.npz                        🟢 Normal  ⚡ DETECTED
  [221/686] chb07_06.npz                        🟢 Normal  ⚡ DETECTED
  [222/686] chb07_07.npz                        🟢 Normal  ⚡ DETECTED
  [223/686] chb07_08.npz                        🟢 Normal  ⚡ DETECTED
  [224/686] chb07_09.npz                        🟢 Normal  ⚡ DETECTED
  [225/686] chb07_10.npz                        🟢 Normal  ⚡ DETECTED
  [226/686] chb07_11.npz                        🟢 Normal  ⚡ DETECTED
  [227/686] chb07_12.npz                        🟢 Normal  ⚡ DETECTED
  [228/686] chb07_13.npz                        🟢 Normal  ⚡ DETECTED
  [229/686] chb07_14.npz                        🟢 Normal  ⚡ DETECTED
  [230/686] chb07_15.npz                        🟢 Normal  ⚡ DETECTED
  [231/686] chb07_16.npz                        🟢 Normal  ⚡ DETECTED
  [232/686] chb07_17.npz                        🟢 Normal  ⚡ DETECTED
  [233/686] chb07_18.npz                        🟢 Normal  ⚡ DETECTED
  [234/686] chb07_19.npz                        🟢 Normal  ⚡ DETECTED
  [235/686] chb08_02.npz                        🟢 Normal  ⚡ DETECTED
  [236/686] chb08_03.npz                        🟢 Normal  ⚡ DETECTED
  [237/686] chb08_04.npz                        🟢 Normal  ⚡ DETECTED
  [238/686] chb08_05.npz                        🟢 Normal  ⚡ DETECTED
  [239/686] chb08_10.npz                        🟢 Normal  ⚡ DETECTED
  [240/686] chb08_11.npz                        🟢 Normal  ⚡ DETECTED
  [241/686] chb08_12.npz                        🟢 Normal  ⚡ DETECTED
  [242/686] chb08_13.npz                        🟢 Normal  ⚡ DETECTED
  [243/686] chb08_14.npz                        🟢 Normal  ⚡ DETECTED
  [244/686] chb08_15.npz                        🟢 Normal  ⚡ DETECTED
  [245/686] chb08_16.npz                        🟢 Normal  ⚡ DETECTED
  [246/686] chb08_17.npz                        🟢 Normal  ⚡ DETECTED
  [247/686] chb08_18.npz                        🟢 Normal  ⚡ DETECTED
  [248/686] chb08_19.npz                        🟢 Normal  ⚡ DETECTED
  [249/686] chb08_20.npz                        🟢 Normal  ⚡ DETECTED
  [250/686] chb08_21.npz                        🟢 Normal  ⚡ DETECTED
  [251/686] chb08_22.npz                        🟢 Normal  ⚡ DETECTED
  [252/686] chb08_23.npz                        🟢 Normal  ⚡ DETECTED
  [253/686] chb08_24.npz                        🟢 Normal  ⚡ DETECTED
  [254/686] chb08_29.npz                        🟢 Normal  ⚡ DETECTED
  [255/686] chb09_01.npz                        🟢 Normal  ⚡ DETECTED
  [256/686] chb09_02.npz                        🟢 Normal  ⚡ DETECTED
  [257/686] chb09_03.npz                        🟢 Normal  ⚡ DETECTED
  [258/686] chb09_04.npz                        🟢 Normal  ⚡ DETECTED
  [259/686] chb09_05.npz                        🟢 Normal  ⚡ DETECTED
  [260/686] chb09_06.npz                        🟢 Normal  ⚡ DETECTED
  [261/686] chb09_07.npz                        🟢 Normal  ⚡ DETECTED
  [262/686] chb09_08.npz                        🟢 Normal  ⚡ DETECTED
  [263/686] chb09_09.npz                        🟢 Normal  ⚡ DETECTED
  [264/686] chb09_10.npz                        🟢 Normal  ⚡ DETECTED
  [265/686] chb09_11.npz                        🟢 Normal  ⚡ DETECTED
  [266/686] chb09_12.npz                        🟢 Normal  ⚡ DETECTED
  [267/686] chb09_13.npz                        🟢 Normal  ⚡ DETECTED
  [268/686] chb09_14.npz                        🟢 Normal  ⚡ DETECTED
  [269/686] chb09_15.npz                        🟢 Normal  ⚡ DETECTED
  [270/686] chb09_16.npz                        🟢 Normal  ⚡ DETECTED
  [271/686] chb09_17.npz                        🟢 Normal  ⚡ DETECTED
  [272/686] chb09_18.npz                        🟢 Normal  ⚡ DETECTED
  [273/686] chb09_19.npz                        🟢 Normal  ⚡ DETECTED
  [274/686] chb10_01.npz                        🟢 Normal  ⚡ DETECTED
  [275/686] chb10_02.npz                        🟢 Normal  ⚡ DETECTED
  [276/686] chb10_03.npz                        🟢 Normal  ⚡ DETECTED
  [277/686] chb10_04.npz                        🟢 Normal  ⚡ DETECTED
  [278/686] chb10_05.npz                        🟢 Normal  ⚡ DETECTED
  [279/686] chb10_06.npz                        🟢 Normal  ⚡ DETECTED
  [280/686] chb10_07.npz                        🟢 Normal  ⚡ DETECTED
  [281/686] chb10_08.npz                        🟢 Normal  ⚡ DETECTED
  [282/686] chb10_12.npz                        🟢 Normal  ⚡ DETECTED
  [283/686] chb10_13.npz                        🟢 Normal  ⚡ DETECTED
  [284/686] chb10_14.npz                        🟢 Normal  ⚡ DETECTED
  [285/686] chb10_15.npz                        🟢 Normal  ⚡ DETECTED
  [286/686] chb10_16.npz                        🟢 Normal  ⚡ DETECTED
  [287/686] chb10_17.npz                        🟢 Normal  ⚡ DETECTED
  [288/686] chb10_18.npz                        🟢 Normal  ⚡ DETECTED
  [289/686] chb10_19.npz                        🟢 Normal  ⚡ DETECTED
  [290/686] chb10_20.npz                        🟢 Normal  ⚡ DETECTED
  [291/686] chb10_21.npz                        🟢 Normal  ⚡ DETECTED
  [292/686] chb10_22.npz                        🟢 Normal  ⚡ DETECTED
  [293/686] chb10_27.npz                        🟢 Normal  ⚡ DETECTED
  [294/686] chb10_28.npz                        🟢 Normal  ⚡ DETECTED
  [295/686] chb10_30.npz                        🟢 Normal  ⚡ DETECTED
  [296/686] chb10_31.npz                        🟢 Normal  ⚡ DETECTED
  [297/686] chb10_38.npz                        🟢 Normal  ⚡ DETECTED
  [298/686] chb10_89.npz                        🟢 Normal  ⚡ DETECTED
  [299/686] chb11_01.npz                        🟢 Normal  ⚡ DETECTED
  [300/686] chb11_02.npz                        🟢 Normal  ⚡ DETECTED
  [301/686] chb11_03.npz                        🟢 Normal  ⚡ DETECTED
  [302/686] chb11_04.npz                        🟢 Normal  ⚡ DETECTED
  [303/686] chb11_05.npz                        🟢 Normal  ⚡ DETECTED
  [304/686] chb11_06.npz                        🟢 Normal  ⚡ DETECTED
  [305/686] chb11_07.npz                        🟢 Normal  ⚡ DETECTED
  [306/686] chb11_08.npz                        🟢 Normal  ⚡ DETECTED
  [307/686] chb11_09.npz                        🟢 Normal  ⚡ DETECTED
  [308/686] chb11_10.npz                        🟢 Normal  ⚡ DETECTED
  [309/686] chb11_11.npz                        🟢 Normal  ⚡ DETECTED
  [310/686] chb11_12.npz                        🟢 Normal  ⚡ DETECTED
  [311/686] chb11_13.npz                        🟢 Normal  ⚡ DETECTED
  [312/686] chb11_14.npz                        🟢 Normal  ⚡ DETECTED
  [313/686] chb11_15.npz                        🟢 Normal  ⚡ DETECTED
  [314/686] chb11_16.npz                        🟢 Normal  ⚡ DETECTED
  [315/686] chb11_17.npz                        🟢 Normal  ⚡ DETECTED
  [316/686] chb11_18.npz                        🟢 Normal  ⚡ DETECTED
  [317/686] chb11_19.npz                        🟢 Normal  ⚡ DETECTED
  [318/686] chb11_24.npz                        🟢 Normal  ⚡ DETECTED
  [319/686] chb11_25.npz                        🟢 Normal  ⚡ DETECTED
  [320/686] chb11_26.npz                        🟢 Normal  ⚡ DETECTED
  [321/686] chb11_27.npz                        🟢 Normal  ⚡ DETECTED
  [322/686] chb11_53.npz                        🟢 Normal  ⚡ DETECTED
  [323/686] chb11_54.npz                        🟢 Normal  ⚡ DETECTED
  [324/686] chb11_55.npz                        🟢 Normal  ⚡ DETECTED
  [325/686] chb11_56.npz                        🟢 Normal  ⚡ DETECTED
  [326/686] chb11_58.npz                        🟢 Normal  ⚡ DETECTED
  [327/686] chb11_60.npz                        🟢 Normal  ⚡ DETECTED
  [328/686] chb11_61.npz                        🟢 Normal  ⚡ DETECTED
  [329/686] chb11_62.npz                        🟢 Normal  ⚡ DETECTED
  [330/686] chb11_63.npz                        🟢 Normal  ⚡ DETECTED
  [331/686] chb11_82.npz                        🟢 Normal  ⚡ DETECTED
  [332/686] chb11_92.npz                        🟢 Normal  ⚡ DETECTED
  [333/686] chb11_99.npz                        🟢 Normal  ⚡ DETECTED
  [334/686] chb12_06.npz                        🟢 Normal  ⚡ DETECTED
  [335/686] chb12_08.npz                        🟢 Normal  ⚡ DETECTED
  [336/686] chb12_09.npz                        🟢 Normal  ⚡ DETECTED
  [337/686] chb12_10.npz                        🟢 Normal  ⚡ DETECTED
  [338/686] chb12_11.npz                        🟢 Normal  ⚡ DETECTED
  [339/686] chb12_19.npz                        🟢 Normal  ⚡ DETECTED
  [340/686] chb12_20.npz                        🟢 Normal  ⚡ DETECTED
  [341/686] chb12_21.npz                        🟢 Normal  ⚡ DETECTED
  [342/686] chb12_23.npz                        🟢 Normal  ⚡ DETECTED
  [343/686] chb12_24.npz                        🟢 Normal  ⚡ DETECTED
  [344/686] chb12_27.npz                        🟢 Normal  ⚡ DETECTED
  [345/686] chb12_28.npz                        🟢 Normal  ⚡ DETECTED
  [346/686] chb12_29.npz                        🟢 Normal  ⚡ DETECTED
  [347/686] chb12_32.npz                        🟢 Normal  ⚡ DETECTED
  [348/686] chb12_33.npz                        🟢 Normal  ⚡ DETECTED
  [349/686] chb12_34.npz                        🟢 Normal  ⚡ DETECTED
  [350/686] chb12_35.npz                        🟢 Normal  ⚡ DETECTED
  [351/686] chb12_36.npz                        🟢 Normal  ⚡ DETECTED
  [352/686] chb12_37.npz                        🟢 Normal  ⚡ DETECTED
  [353/686] chb12_38.npz                        🟢 Normal  ⚡ DETECTED
  [354/686] chb12_39.npz                        🟢 Normal  ⚡ DETECTED
  [355/686] chb12_40.npz                        🟢 Normal  ⚡ DETECTED
  [356/686] chb12_41.npz                        🟢 Normal  ⚡ DETECTED
  [357/686] chb12_42.npz                        🟢 Normal  ⚡ DETECTED
  [358/686] chb13_02.npz                        🟢 Normal  ⚡ DETECTED
  [359/686] chb13_03.npz                        🟢 Normal  ⚡ DETECTED
  [360/686] chb13_04.npz                        🟢 Normal  ⚡ DETECTED
17:48:38 | WARNING  |   ✗ Failed chb13_05.npz: No snn_features in output/spikes/chb13/chb13_05.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_06.npz: No snn_features in output/spikes/chb13/chb13_06.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_07.npz: No snn_features in output/spikes/chb13/chb13_07.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_08.npz: No snn_features in output/spikes/chb13/chb13_08.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_09.npz: No snn_features in output/spikes/chb13/chb13_09.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_10.npz: No snn_features in output/spikes/chb13/chb13_10.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_11.npz: No snn_features in output/spikes/chb13/chb13_11.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_12.npz: No snn_features in output/spikes/chb13/chb13_12.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_13.npz: No snn_features in output/spikes/chb13/chb13_13.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_14.npz: No snn_features in output/spikes/chb13/chb13_14.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_15.npz: No snn_features in output/spikes/chb13/chb13_15.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_16.npz: No snn_features in output/spikes/chb13/chb13_16.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:38 | WARNING  |   ✗ Failed chb13_18.npz: No snn_features in output/spikes/chb13/chb13_18.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
  [374/686] chb13_19.npz                        🟢 Normal  ⚡ DETECTED
  [375/686] chb13_21.npz                        🟢 Normal  ⚡ DETECTED
  [376/686] chb13_22.npz                        🟢 Normal  ⚡ DETECTED
17:48:44 | WARNING  |   ✗ Failed chb13_24.npz: No snn_features in output/spikes/chb13/chb13_24.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:44 | WARNING  |   ✗ Failed chb13_30.npz: No snn_features in output/spikes/chb13/chb13_30.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:44 | WARNING  |   ✗ Failed chb13_36.npz: No snn_features in output/spikes/chb13/chb13_36.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:44 | WARNING  |   ✗ Failed chb13_37.npz: No snn_features in output/spikes/chb13/chb13_37.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:44 | WARNING  |   ✗ Failed chb13_38.npz: No snn_features in output/spikes/chb13/chb13_38.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:44 | WARNING  |   ✗ Failed chb13_39.npz: No snn_features in output/spikes/chb13/chb13_39.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:44 | WARNING  |   ✗ Failed chb13_40.npz: No snn_features in output/spikes/chb13/chb13_40.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:48:44 | WARNING  |   ✗ Failed chb13_47.npz: No snn_features in output/spikes/chb13/chb13_47.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
  [385/686] chb13_55.npz                        🟢 Normal  ⚡ DETECTED
  [386/686] chb13_56.npz                        🟢 Normal  ⚡ DETECTED
  [387/686] chb13_58.npz                        🟢 Normal  ⚡ DETECTED
  [388/686] chb13_59.npz                        🟢 Normal  ⚡ DETECTED
  [389/686] chb13_60.npz                        🟢 Normal  ⚡ DETECTED
  [390/686] chb13_62.npz                        🟢 Normal  ⚡ DETECTED
  [391/686] chb14_01.npz                        🟢 Normal  ⚡ DETECTED
  [392/686] chb14_02.npz                        🟢 Normal  ⚡ DETECTED
  [393/686] chb14_03.npz                        🟢 Normal  ⚡ DETECTED
  [394/686] chb14_04.npz                        🟢 Normal  ⚡ DETECTED
  [395/686] chb14_06.npz                        🟢 Normal  ⚡ DETECTED
  [396/686] chb14_07.npz                        🟢 Normal  ⚡ DETECTED
  [397/686] chb14_11.npz                        🟢 Normal  ⚡ DETECTED
  [398/686] chb14_12.npz                        🟢 Normal  ⚡ DETECTED
  [399/686] chb14_13.npz                        🟢 Normal  ⚡ DETECTED
  [400/686] chb14_14.npz                        🟢 Normal  ⚡ DETECTED
  [401/686] chb14_16.npz                        🟢 Normal  ⚡ DETECTED
  [402/686] chb14_17.npz                        🟢 Normal  ⚡ DETECTED
  [403/686] chb14_18.npz                        🟢 Normal  ⚡ DETECTED
  [404/686] chb14_19.npz                        🟢 Normal  ⚡ DETECTED
  [405/686] chb14_20.npz                        🟢 Normal  ⚡ DETECTED
  [406/686] chb14_22.npz                        🟢 Normal  ⚡ DETECTED
  [407/686] chb14_24.npz                        🟢 Normal  ⚡ DETECTED
  [408/686] chb14_25.npz                        🟢 Normal  ⚡ DETECTED
  [409/686] chb14_26.npz                        🟢 Normal  ⚡ DETECTED
  [410/686] chb14_27.npz                        🟢 Normal  ⚡ DETECTED
  [411/686] chb14_29.npz                        🟢 Normal  ⚡ DETECTED
  [412/686] chb14_30.npz                        🟢 Normal  ⚡ DETECTED
  [413/686] chb14_32.npz                        🟢 Normal  ⚡ DETECTED
  [414/686] chb14_37.npz                        🟢 Normal  ⚡ DETECTED
  [415/686] chb14_39.npz                        🟢 Normal  ⚡ DETECTED
  [416/686] chb14_42.npz                        🟢 Normal  ⚡ DETECTED
  [417/686] chb15_01.npz                        🟢 Normal  ⚡ DETECTED
  [418/686] chb15_02.npz                        🟢 Normal  ⚡ DETECTED
  [419/686] chb15_03.npz                        🟢 Normal  ⚡ DETECTED
  [420/686] chb15_04.npz                        🟢 Normal  ⚡ DETECTED
  [421/686] chb15_05.npz                        🟢 Normal  ⚡ DETECTED
  [422/686] chb15_06.npz                        🟢 Normal  ⚡ DETECTED
  [423/686] chb15_07.npz                        🟢 Normal  ⚡ DETECTED
  [424/686] chb15_08.npz                        🟢 Normal  ⚡ DETECTED
  [425/686] chb15_09.npz                        🟢 Normal  ⚡ DETECTED
  [426/686] chb15_10.npz                        🟢 Normal  ⚡ DETECTED
  [427/686] chb15_11.npz                        🟢 Normal  ⚡ DETECTED
  [428/686] chb15_12.npz                        🟢 Normal  ⚡ DETECTED
  [429/686] chb15_13.npz                        🟢 Normal  ⚡ DETECTED
  [430/686] chb15_14.npz                        🟢 Normal  ⚡ DETECTED
  [431/686] chb15_15.npz                        🟢 Normal  ⚡ DETECTED
  [432/686] chb15_16.npz                        🟢 Normal  ⚡ DETECTED
  [433/686] chb15_17.npz                        🟢 Normal  ⚡ DETECTED
  [434/686] chb15_19.npz                        🟢 Normal  ⚡ DETECTED
  [435/686] chb15_20.npz                        🟢 Normal  ⚡ DETECTED
  [436/686] chb15_22.npz                        🟢 Normal  ⚡ DETECTED
  [437/686] chb15_26.npz                        🟢 Normal  ⚡ DETECTED
  [438/686] chb15_28.npz                        🟢 Normal  ⚡ DETECTED
  [439/686] chb15_29.npz                        🟢 Normal  ⚡ DETECTED
  [440/686] chb15_30.npz                        🟢 Normal  ⚡ DETECTED
  [441/686] chb15_31.npz                        🟢 Normal  ⚡ DETECTED
  [442/686] chb15_32.npz                        🟢 Normal  ⚡ DETECTED
  [443/686] chb15_33.npz                        🟢 Normal  ⚡ DETECTED
  [444/686] chb15_35.npz                        🟢 Normal  ⚡ DETECTED
  [445/686] chb15_37.npz                        🟢 Normal  ⚡ DETECTED
  [446/686] chb15_40.npz                        🟢 Normal  ⚡ DETECTED
  [447/686] chb15_45.npz                        🟢 Normal  ⚡ DETECTED
  [448/686] chb15_46.npz                        🟢 Normal  ⚡ DETECTED
  [449/686] chb15_49.npz                        🟢 Normal  ⚡ DETECTED
  [450/686] chb15_50.npz                        🟢 Normal  ⚡ DETECTED
  [451/686] chb15_51.npz                        🟢 Normal  ⚡ DETECTED
  [452/686] chb15_52.npz                        🟢 Normal  ⚡ DETECTED
  [453/686] chb15_54.npz                        🟢 Normal  ⚡ DETECTED
  [454/686] chb15_61.npz                        🟢 Normal  ⚡ DETECTED
  [455/686] chb15_62.npz                        🟢 Normal  ⚡ DETECTED
  [456/686] chb15_63.npz                        🟢 Normal  ⚡ DETECTED
  [457/686] chb16_01.npz                        🟢 Normal  ⚡ DETECTED
  [458/686] chb16_02.npz                        🟢 Normal  ⚡ DETECTED
  [459/686] chb16_03.npz                        🟢 Normal  ⚡ DETECTED
  [460/686] chb16_04.npz                        🟢 Normal  ⚡ DETECTED
  [461/686] chb16_05.npz                        🟢 Normal  ⚡ DETECTED
  [462/686] chb16_06.npz                        🟢 Normal  ⚡ DETECTED
  [463/686] chb16_07.npz                        🟢 Normal  ⚡ DETECTED
  [464/686] chb16_08.npz                        🟢 Normal  ⚡ DETECTED
  [465/686] chb16_09.npz                        🟢 Normal  ⚡ DETECTED
  [466/686] chb16_10.npz                        🟢 Normal  ⚡ DETECTED
  [467/686] chb16_11.npz                        🟢 Normal  ⚡ DETECTED
  [468/686] chb16_12.npz                        🟢 Normal  ⚡ DETECTED
  [469/686] chb16_13.npz                        🟢 Normal  ⚡ DETECTED
  [470/686] chb16_14.npz                        🟢 Normal  ⚡ DETECTED
  [471/686] chb16_15.npz                        🟢 Normal  ⚡ DETECTED
  [472/686] chb16_16.npz                        🟢 Normal  ⚡ DETECTED
  [473/686] chb16_17.npz                        🟢 Normal  ⚡ DETECTED
17:51:43 | WARNING  |   ✗ Failed chb16_18.npz: No snn_features in output/spikes/chb16/chb16_18.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:51:43 | WARNING  |   ✗ Failed chb16_19.npz: No snn_features in output/spikes/chb16/chb16_19.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
  [476/686] chb17a_03.npz                       🟢 Normal  ⚡ DETECTED
  [477/686] chb17a_04.npz                       🟢 Normal  ⚡ DETECTED
  [478/686] chb17a_05.npz                       🟢 Normal  ⚡ DETECTED
  [479/686] chb17a_06.npz                       🟢 Normal  ⚡ DETECTED
  [480/686] chb17a_08.npz                       🟢 Normal  ⚡ DETECTED
  [481/686] chb17b_57.npz                       🟢 Normal  ⚡ DETECTED
  [482/686] chb17b_58.npz                       🟢 Normal  ⚡ DETECTED
  [483/686] chb17b_59.npz                       🟢 Normal  ⚡ DETECTED
  [484/686] chb17b_60.npz                       🟢 Normal  ⚡ DETECTED
  [485/686] chb17b_63.npz                       🟢 Normal  ⚡ DETECTED
  [486/686] chb17b_67.npz                       🟢 Normal  ⚡ DETECTED
  [487/686] chb17b_68.npz                       🟢 Normal  ⚡ DETECTED
  [488/686] chb17b_69.npz                       🟢 Normal  ⚡ DETECTED
  [489/686] chb17c_02.npz                       🟢 Normal  ⚡ DETECTED
  [490/686] chb17c_03.npz                       🟢 Normal  ⚡ DETECTED
  [491/686] chb17c_04.npz                       🟢 Normal  ⚡ DETECTED
  [492/686] chb17c_05.npz                       🟢 Normal  ⚡ DETECTED
  [493/686] chb17c_06.npz                       🟢 Normal  ⚡ DETECTED
  [494/686] chb17c_07.npz                       🟢 Normal  ⚡ DETECTED
  [495/686] chb17c_08.npz                       🟢 Normal  ⚡ DETECTED
17:52:21 | WARNING  |   ✗ Failed chb17c_13.npz: No snn_features in output/spikes/chb17/chb17c_13.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
17:52:21 | WARNING  |   ✗ Failed chb18_01.npz: No snn_features in output/spikes/chb18/chb18_01.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
  [498/686] chb18_02.npz                        🟢 Normal  ⚡ DETECTED
  [499/686] chb18_03.npz                        🟢 Normal  ⚡ DETECTED
  [500/686] chb18_04.npz                        🟢 Normal  ⚡ DETECTED
  [501/686] chb18_05.npz                        🟢 Normal  ⚡ DETECTED
  [502/686] chb18_06.npz                        🟢 Normal  ⚡ DETECTED
  [503/686] chb18_07.npz                        🟢 Normal  ⚡ DETECTED
  [504/686] chb18_08.npz                        🟢 Normal  ⚡ DETECTED
  [505/686] chb18_09.npz                        🟢 Normal  ⚡ DETECTED
  [506/686] chb18_10.npz                        🟢 Normal  ⚡ DETECTED
  [507/686] chb18_11.npz                        🟢 Normal  ⚡ DETECTED
  [508/686] chb18_12.npz                        🟢 Normal  ⚡ DETECTED
  [509/686] chb18_13.npz                        🟢 Normal  ⚡ DETECTED
  [510/686] chb18_14.npz                        🟢 Normal  ⚡ DETECTED
  [511/686] chb18_15.npz                        🟢 Normal  ⚡ DETECTED
  [512/686] chb18_16.npz                        🟢 Normal  ⚡ DETECTED
  [513/686] chb18_17.npz                        🟢 Normal  ⚡ DETECTED
  [514/686] chb18_18.npz                        🟢 Normal  ⚡ DETECTED
  [515/686] chb18_19.npz                        🟢 Normal  ⚡ DETECTED
  [516/686] chb18_20.npz                        🟢 Normal  ⚡ DETECTED
  [517/686] chb18_21.npz                        🟢 Normal  ⚡ DETECTED
  [518/686] chb18_22.npz                        🟢 Normal  ⚡ DETECTED
  [519/686] chb18_23.npz                        🟢 Normal  ⚡ DETECTED
  [520/686] chb18_24.npz                        🟢 Normal  ⚡ DETECTED
  [521/686] chb18_25.npz                        🟢 Normal  ⚡ DETECTED
  [522/686] chb18_26.npz                        🟢 Normal  ⚡ DETECTED
  [523/686] chb18_27.npz                        🟢 Normal  ⚡ DETECTED
  [524/686] chb18_28.npz                        🟢 Normal  ⚡ DETECTED
  [525/686] chb18_29.npz                        🟢 Normal  ⚡ DETECTED
  [526/686] chb18_30.npz                        🟢 Normal  ⚡ DETECTED
  [527/686] chb18_31.npz                        🟢 Normal  ⚡ DETECTED
  [528/686] chb18_32.npz                        🟢 Normal  ⚡ DETECTED
  [529/686] chb18_33.npz                        🟢 Normal  ⚡ DETECTED
  [530/686] chb18_34.npz                        🟢 Normal  ⚡ DETECTED
  [531/686] chb18_35.npz                        🟢 Normal  ⚡ DETECTED
  [532/686] chb18_36.npz                        🟢 Normal  ⚡ DETECTED
17:53:26 | WARNING  |   ✗ Failed chb19_01.npz: No snn_features in output/spikes/chb19/chb19_01.npz. Keys: ['spikes', 'binary_labels', 'preictal_labels', 'seizure_intervals', 'spike_method', 'sampling_rate', 'n_channels']
  [534/686] chb19_02.npz                        🟢 Normal  ⚡ DETECTED
  [535/686] chb19_03.npz                        🟢 Normal  ⚡ DETECTED
  [536/686] chb19_04.npz                        🟢 Normal  ⚡ DETECTED
  [537/686] chb19_05.npz                        🟢 Normal  ⚡ DETECTED
  [538/686] chb19_06.npz                        🟢 Normal  ⚡ DETECTED
  [539/686] chb19_07.npz                        🟢 Normal  ⚡ DETECTED
  [540/686] chb19_08.npz                        🟢 Normal  ⚡ DETECTED
  [541/686] chb19_09.npz                        🟢 Normal  ⚡ DETECTED
  [542/686] chb19_10.npz                        🟢 Normal  ⚡ DETECTED
  [543/686] chb19_11.npz                        🟢 Normal  ⚡ DETECTED
  [544/686] chb19_12.npz                        🟢 Normal  ⚡ DETECTED
  [545/686] chb19_13.npz                        🟢 Normal  ⚡ DETECTED
  [546/686] chb19_14.npz                        🟢 Normal  ⚡ DETECTED
  [547/686] chb19_15.npz                        🟢 Normal  ⚡ DETECTED
  [548/686] chb19_16.npz                        🟢 Normal  ⚡ DETECTED
  [549/686] chb19_17.npz                        🟢 Normal  ⚡ DETECTED
  [550/686] chb19_18.npz                        🟢 Normal  ⚡ DETECTED
  [551/686] chb19_19.npz                        🟢 Normal  ⚡ DETECTED
  [552/686] chb19_20.npz                        🟢 Normal  ⚡ DETECTED
  [553/686] chb19_21.npz                        🟢 Normal  ⚡ DETECTED
  [554/686] chb19_22.npz                        🟢 Normal  ⚡ DETECTED
  [555/686] chb19_23.npz                        🟢 Normal  ⚡ DETECTED
  [556/686] chb19_24.npz                        🟢 Normal  ⚡ DETECTED
  [557/686] chb19_25.npz                        🟢 Normal  ⚡ DETECTED
  [558/686] chb19_26.npz                        🟢 Normal  ⚡ DETECTED
  [559/686] chb19_27.npz                        🟢 Normal  ⚡ DETECTED
  [560/686] chb19_28.npz                        🟢 Normal  ⚡ DETECTED
  [561/686] chb19_29.npz                        🟢 Normal  ⚡ DETECTED
  [562/686] chb19_30.npz                        🟢 Normal  ⚡ DETECTED
  [563/686] chb20_01.npz                        🟢 Normal  ⚡ DETECTED
  [564/686] chb20_02.npz                        🟢 Normal  ⚡ DETECTED
  [565/686] chb20_03.npz                        🟢 Normal  ⚡ DETECTED
  [566/686] chb20_04.npz                        🟢 Normal  ⚡ DETECTED
  [567/686] chb20_05.npz                        🟢 Normal  ⚡ DETECTED
  [568/686] chb20_06.npz                        🟢 Normal  ⚡ DETECTED
  [569/686] chb20_07.npz                        🟢 Normal  ⚡ DETECTED
  [570/686] chb20_08.npz                        🟢 Normal  ⚡ DETECTED
  [571/686] chb20_11.npz                        🟢 Normal  ⚡ DETECTED
  [572/686] chb20_12.npz                        🟢 Normal  ⚡ DETECTED
  [573/686] chb20_13.npz                        🟢 Normal  ⚡ DETECTED
  [574/686] chb20_14.npz                        🟢 Normal  ⚡ DETECTED
  [575/686] chb20_15.npz                        🟢 Normal  ⚡ DETECTED
  [576/686] chb20_16.npz                        🟢 Normal  ⚡ DETECTED
  [577/686] chb20_17.npz                        🟢 Normal  ⚡ DETECTED
  [578/686] chb20_21.npz                        🟢 Normal  ⚡ DETECTED
  [579/686] chb20_22.npz                        🟢 Normal  ⚡ DETECTED
  [580/686] chb20_23.npz                        🟢 Normal  ⚡ DETECTED
  [581/686] chb20_25.npz                        🟢 Normal  ⚡ DETECTED
  [582/686] chb20_26.npz                        🟢 Normal  ⚡ DETECTED
  [583/686] chb20_27.npz                        🟢 Normal  ⚡ DETECTED
  [584/686] chb20_28.npz                        🟢 Normal  ⚡ DETECTED
  [585/686] chb20_29.npz                        🟢 Normal  ⚡ DETECTED
  [586/686] chb20_30.npz                        🟢 Normal  ⚡ DETECTED
  [587/686] chb20_31.npz                        🟢 Normal  ⚡ DETECTED
  [588/686] chb20_34.npz                        🟢 Normal  ⚡ DETECTED
  [589/686] chb20_59.npz                        🟢 Normal  ⚡ DETECTED
  [590/686] chb20_60.npz                        🟢 Normal  ⚡ DETECTED
  [591/686] chb20_68.npz                        🟢 Normal  ⚡ DETECTED
  [592/686] chb21_01.npz                        🟢 Normal  ⚡ DETECTED
  [593/686] chb21_02.npz                        🟢 Normal  ⚡ DETECTED
  [594/686] chb21_03.npz                        🟢 Normal  ⚡ DETECTED
  [595/686] chb21_04.npz                        🟢 Normal  ⚡ DETECTED
  [596/686] chb21_05.npz                        🟢 Normal  ⚡ DETECTED
  [597/686] chb21_06.npz                        🟢 Normal  ⚡ DETECTED
  [598/686] chb21_07.npz                        🟢 Normal  ⚡ DETECTED
  [599/686] chb21_08.npz                        🟢 Normal  ⚡ DETECTED
  [600/686] chb21_09.npz                        🟢 Normal  ⚡ DETECTED
  [601/686] chb21_10.npz                        🟢 Normal  ⚡ DETECTED
  [602/686] chb21_11.npz                        🟢 Normal  ⚡ DETECTED
  [603/686] chb21_12.npz                        🟢 Normal  ⚡ DETECTED
  [604/686] chb21_13.npz                        🟢 Normal  ⚡ DETECTED
  [605/686] chb21_14.npz                        🟢 Normal  ⚡ DETECTED
  [606/686] chb21_15.npz                        🟢 Normal  ⚡ DETECTED
  [607/686] chb21_16.npz                        🟢 Normal  ⚡ DETECTED
  [608/686] chb21_17.npz                        🟢 Normal  ⚡ DETECTED
  [609/686] chb21_18.npz                        🟢 Normal  ⚡ DETECTED
  [610/686] chb21_19.npz                        🟢 Normal  ⚡ DETECTED
  [611/686] chb21_20.npz                        🟢 Normal  ⚡ DETECTED
  [612/686] chb21_21.npz                        🟢 Normal  ⚡ DETECTED
  [613/686] chb21_22.npz                        🟢 Normal  ⚡ DETECTED
  [614/686] chb21_23.npz                        🟢 Normal  ⚡ DETECTED
  [615/686] chb21_24.npz                        🟢 Normal  ⚡ DETECTED
  [616/686] chb21_25.npz                        🟢 Normal  ⚡ DETECTED
  [617/686] chb21_26.npz                        🟢 Normal  ⚡ DETECTED
  [618/686] chb21_27.npz                        🟢 Normal  ⚡ DETECTED
  [619/686] chb21_28.npz                        🟢 Normal  ⚡ DETECTED
  [620/686] chb21_29.npz                        🟢 Normal  ⚡ DETECTED
  [621/686] chb21_30.npz                        🟢 Normal  ⚡ DETECTED
  [622/686] chb21_31.npz                        🟢 Normal  ⚡ DETECTED
  [623/686] chb21_32.npz                        🟢 Normal  ⚡ DETECTED
  [624/686] chb21_33.npz                        🟢 Normal  ⚡ DETECTED
  [625/686] chb22_01.npz                        🟢 Normal  ⚡ DETECTED
  [626/686] chb22_02.npz                        🟢 Normal  ⚡ DETECTED
  [627/686] chb22_03.npz                        🟢 Normal  ⚡ DETECTED
  [628/686] chb22_04.npz                        🟢 Normal  ⚡ DETECTED
  [629/686] chb22_05.npz                        🟢 Normal  ⚡ DETECTED
  [630/686] chb22_06.npz                        🟢 Normal  ⚡ DETECTED
  [631/686] chb22_07.npz                        🟢 Normal  ⚡ DETECTED
  [632/686] chb22_08.npz                        🟢 Normal  ⚡ DETECTED
  [633/686] chb22_09.npz                        🟢 Normal  ⚡ DETECTED
  [634/686] chb22_10.npz                        🟢 Normal  ⚡ DETECTED
  [635/686] chb22_11.npz                        🟢 Normal  ⚡ DETECTED
  [636/686] chb22_15.npz                        🟢 Normal  ⚡ DETECTED
  [637/686] chb22_16.npz                        🟢 Normal  ⚡ DETECTED
  [638/686] chb22_17.npz                        🟢 Normal  ⚡ DETECTED
  [639/686] chb22_18.npz                        🟢 Normal  ⚡ DETECTED
  [640/686] chb22_19.npz                        🟢 Normal  ⚡ DETECTED
  [641/686] chb22_20.npz                        🟢 Normal  ⚡ DETECTED
  [642/686] chb22_21.npz                        🟢 Normal  ⚡ DETECTED
  [643/686] chb22_22.npz                        🟢 Normal  ⚡ DETECTED
  [644/686] chb22_23.npz                        🟢 Normal  ⚡ DETECTED
  [645/686] chb22_24.npz                        🟢 Normal  ⚡ DETECTED
  [646/686] chb22_25.npz                        🟢 Normal  ⚡ DETECTED
  [647/686] chb22_26.npz                        🟢 Normal  ⚡ DETECTED
  [648/686] chb22_27.npz                        🟢 Normal  ⚡ DETECTED
  [649/686] chb22_28.npz                        🟢 Normal  ⚡ DETECTED
  [650/686] chb22_29.npz                        🟢 Normal  ⚡ DETECTED
  [651/686] chb22_30.npz                        🟢 Normal  ⚡ DETECTED
  [652/686] chb22_38.npz                        🟢 Normal  ⚡ DETECTED
  [653/686] chb22_51.npz                        🟢 Normal  ⚡ DETECTED
  [654/686] chb22_54.npz                        🟢 Normal  ⚡ DETECTED
  [655/686] chb22_77.npz                        🟢 Normal  ⚡ DETECTED
  [656/686] chb23_06.npz                        🟢 Normal  ⚡ DETECTED
  [657/686] chb23_07.npz                        🟢 Normal  ⚡ DETECTED
  [658/686] chb23_08.npz                        🟢 Normal  ⚡ DETECTED
  [659/686] chb23_09.npz                        🟢 Normal  ⚡ DETECTED
  [660/686] chb23_10.npz                        🟢 Normal  ⚡ DETECTED
  [661/686] chb23_16.npz                        🟢 Normal  ⚡ DETECTED
  [662/686] chb23_17.npz                        🟢 Normal  ⚡ DETECTED
  [663/686] chb23_19.npz                        🟢 Normal  ⚡ DETECTED
  [664/686] chb23_20.npz                        🟢 Normal  ⚡ DETECTED
  [665/686] chb24_01.npz                        🟢 Normal  ⚡ DETECTED
  [666/686] chb24_02.npz                        🟢 Normal  ⚡ DETECTED
  [667/686] chb24_03.npz                        🟢 Normal  ⚡ DETECTED
  [668/686] chb24_04.npz                        🟢 Normal  ⚡ DETECTED
  [669/686] chb24_05.npz                        🟢 Normal  ⚡ DETECTED
  [670/686] chb24_06.npz                        🟢 Normal  ⚡ DETECTED
  [671/686] chb24_07.npz                        🟢 Normal  ⚡ DETECTED
  [672/686] chb24_08.npz                        🟢 Normal  ⚡ DETECTED
  [673/686] chb24_09.npz                        🟢 Normal  ⚡ DETECTED
  [674/686] chb24_10.npz                        🟢 Normal  ⚡ DETECTED
  [675/686] chb24_11.npz                        🟢 Normal  ⚡ DETECTED
  [676/686] chb24_12.npz                        🟢 Normal  ⚡ DETECTED
  [677/686] chb24_13.npz                        🟢 Normal  ⚡ DETECTED
  [678/686] chb24_14.npz                        🟢 Normal  ⚡ DETECTED
  [679/686] chb24_15.npz                        🟢 Normal  ⚡ DETECTED
  [680/686] chb24_16.npz                        🟢 Normal  ⚡ DETECTED
  [681/686] chb24_17.npz                        🟢 Normal  ⚡ DETECTED
  [682/686] chb24_18.npz                        🟢 Normal  ⚡ DETECTED
  [683/686] chb24_19.npz                        🟢 Normal  ⚡ DETECTED
  [684/686] chb24_20.npz                        🟢 Normal  ⚡ DETECTED
  [685/686] chb24_21.npz                        🟢 Normal  ⚡ DETECTED
  [686/686] chb24_22.npz                        🟢 Normal  ⚡ DETECTED
17:58:35 | INFO     | 
Aggregating metrics across all files...
17:58:41 | INFO     | Saved model comparison chart: test_plots/model_comparison.png
17:58:43 | INFO     | Saved ROC curve: test_plots/roc_autoencoder.png
17:58:45 | INFO     | Saved ROC curve: test_plots/roc_lstm.png
17:58:48 | INFO     | Saved ROC curve: test_plots/roc_cnn.png
17:58:52 | INFO     | Saved ROC curve: test_plots/roc_transformer.png
17:58:54 | INFO     | Saved ROC curve: test_plots/roc_ensemble.png
17:58:54 | INFO     | Saved subject summary: test_plots/subject_detection_rate.png


█████████████████████████████████████████████████████████████████
█  TEST RESULTS — AGGREGATED ACROSS ALL FILES
█████████████████████████████████████████████████████████████████

Model             Accuracy         F1     Recall    AUC-ROC  Precision
─────────────────────────────────────────────────────────────────
autoencoder         0.8679     0.0196     0.3993     0.7367     0.0101
lstm                0.9594     0.0776     0.5158     0.8140     0.0420
cnn                 0.7209     0.0220     0.9466     0.9519     0.0111
transformer         0.8645     0.0410     0.8750     0.9279     0.0210
ensemble            0.9081     0.0589     0.8672     0.9452     0.0305
─────────────────────────────────────────────────────────────────

  Files tested: 660 | Failed: 26
  Plots saved : test_plots