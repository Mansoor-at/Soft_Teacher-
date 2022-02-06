_base_ = "base.py"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        sup=dict(
            ann_file="/raid/home/A01753093/SoftTeacher/tools/dataset/data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="/raid/home/A01753093/SoftTeacher/tools/dataset/data/coco/train2017/",
        ),
        unsup=dict(
            #ann_file="/raid/home/A01753093/SoftTeacher/tools/dataset/data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            ann_file="/raid/home/A01753093/SoftTeacher/tools/dataset/data/coco/annotations/instances_unlabeled2017.json",
            img_prefix="/raid/home/A01753093/SoftTeacher/tools/dataset/data/coco/unlabeled2017/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="pre_release",
                name="${cfg_name}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
