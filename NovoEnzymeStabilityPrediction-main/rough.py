# load weights from custom saved model .h5 and predict output

model = load_model('model_esp.h5')
model.load_weights('model_esp.h5')