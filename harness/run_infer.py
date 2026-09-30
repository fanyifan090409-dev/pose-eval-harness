"""T2 推断层：单次 val(save_json=True) 双算。

跑一次 model.val() 同时产出：
- official.json：ultralytics 官方全量指标（门禁 1 对拍源）
- preds.json：save_json 的预测结果（M1/M2 引擎输入，格式同 ultralytics save_json）

用法：
    python harness/run_infer.py --model round5-3 --dataset crowdpose-val
    python harness/run_infer.py --model round5-3 --dataset posetrack-val --device 0,1

待实现（T2）：
- 从 configs/models.yaml、configs/datasets.yaml 读路径
- COCO val yaml 生成（person_keypoints_val2017.json → YOLO 标签，官方 5000 张口径）
- CrowdPose data.yaml names 遗留修复（7 类 → person）
- device 参数留给 M3 双卡分片
"""


def main():
    raise NotImplementedError("T2 实现")


if __name__ == "__main__":
    main()
