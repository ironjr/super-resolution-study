SET SCHEDULE=schedule.json
SET TRY_NEW=True
SET USE_TENSORBOARD=True
SET USE_GPU=True
SET NUM_TRAINING_SET=50000
SET BATCH_SIZE=64
SET PRINT_EVERY=100

python main.py^
    --mode='train'^
    --schedule=%SCHEDULE%^
    --use-tb=%USE_TENSORBOARD%^
    --use-gpu=%USE_GPU%^
    --try-new=%TRY_NEW%^
    --num-train=%NUM_TRAINING_SET%^
    --batch-size=%BATCH_SIZE%^
    --print-every=%PRINT_EVERY%
