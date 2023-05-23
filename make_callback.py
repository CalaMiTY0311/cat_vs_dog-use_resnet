from datetime import datetime
import os
import keras

save_dir = './my_log'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

    
project_name = 'dog_cat_CNN_model'

def save_file():
    time = datetime.today()
    yy = time.year
    mon = time.month
    dd = time.day
    hh = time.hour
    mm = time.minute
    sec = time.second
    time_name = str(yy) +  str(mon) + str(dd) + str(hh) + str(mm) +  str(sec) +'_my_' + project_name + '_model.h5'
    file_name = os.path.join(save_dir,time_name)
    return file_name

callbacks = [
    
    keras.callbacks.TensorBoard(
    log_dir = save_dir,
    write_graph=True,
    write_images=True
    ),
    
    keras.callbacks.EarlyStopping(
    monitor = 'val_acc',
        patience=10,
    ),
    keras.callbacks.ModelCheckpoint(
    filepath= save_file(),
    monitor = 'val_loss',
    save_best_only = True,
    )
]