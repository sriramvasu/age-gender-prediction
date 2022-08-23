from utils import *
from model import *
from loader import *

def validate(valid_loader, vggface_model, gender_model, writer, epoch, age_model):	
	batch_confmat=np.zeros([2,2])
	age_confmat=np.zeros([14,14])
	for ij, item in enumerate(valid_loader):
		out=vggface_model(input=item[0])
		gender_out=gender_model.predict(out[0])
		age_out=age_model.predict(out[0])
		batch_confmat+=confusion_matrix(item[1], tf.argmax(gender_out, axis=1).numpy(), labels=range(2))
		age_confmat+=confusion_matrix(item[3], tf.argmax(age_out, axis=1).numpy(), labels=range(14))
		print('Validation epoch: %d, iteration: %d, confmat: %s, male_acc: %f, female_acc: %f, age_acc:%f'%(epoch, ij, batch_confmat, batch_confmat[0,0]/np.sum(batch_confmat[0,:]), batch_confmat[1,1]/np.sum(batch_confmat[1,:]), np.sum(np.diag(age_confmat))/np.sum(age_confmat)))
	with writer.as_default():
		tf.summary.scalar('valid/male_accuracy', batch_confmat[0,0]/np.sum(batch_confmat[0,:]), step=epoch)
		tf.summary.scalar('valid/female_accuracy', batch_confmat[1,1]/np.sum(batch_confmat[1,:]), step=epoch)
		tf.summary.scalar('train/age_accuracy', np.sum(np.diag(age_confmat))/np.sum(age_confmat), step=epoch)

def train(train_loader, valid_loader, vggface_model, gender_model, gender_op, criterion, writer, age_model):
	global_step=0
	for epoch in range(100):
		if(epoch%2==0 and epoch>=1):
			gender_model.save('./models/gendermodel_%d_%s'%(epoch, prefix))
			age_model.save('./models/agemodel_%d_%s'%(epoch, prefix))
		if(epoch%2==0 and epoch>=1):
			validate(valid_loader, vggface_model, gender_model, writer, epoch, age_model)
		batch_confmat=np.zeros([2,2])
		age_confmat=np.zeros([14,14])
		for ij,item in enumerate(train_loader):
			out=vggface_model(input=item[0])
			print(out[0].shape)
			gender_loss=gender_model.train_on_batch(out[0], item[1])
			age_loss=age_model.train_on_batch(out[0], item[3])
			gender_out=gender_model.predict(out[0])
			age_out=age_model.predict(out[0])
			batch_confmat+=confusion_matrix(item[1], tf.argmax(gender_out, axis=1).numpy(), labels=range(2))	
			age_confmat+=confusion_matrix(item[3], tf.argmax(age_out, axis=1).numpy(), labels=range(14))		
			print('Training epoch: %d, iteration: %d, Loss:%f, confmat: %s, male_acc: %f, female_acc: %f, age_acc:%f'%(epoch, ij, gender_loss, batch_confmat, batch_confmat[0,0]/np.sum(batch_confmat[0,:]), batch_confmat[1,1]/np.sum(batch_confmat[1,:]), np.sum(np.diag(age_confmat))/np.sum(age_confmat)))
			with writer.as_default():
				tf.summary.scalar('train/gender_loss', gender_loss, step=global_step)
				tf.summary.scalar('train/age_loss', age_loss, step=global_step)
				tf.summary.scalar('train/male_accuracy', batch_confmat[0,0]/np.sum(batch_confmat[0,:]), step=global_step)
				tf.summary.scalar('train/female_accuracy', batch_confmat[1,1]/np.sum(batch_confmat[1,:]), step=global_step)
				tf.summary.scalar('train/age_accuracy', np.sum(np.diag(age_confmat))/np.sum(age_confmat), step=global_step)
				global_step+=1

def test_model(test_loader, vggface_model, gender_model, age_model):
	vggface_model=tf.saved_model.load('./vggface_tensorflow.pb')
	gender_model=tf.keras.models.load_model('./gender_model_epoch74')
	age_model=tf.keras.models.load_model('./agemodel_54')
	batch_confmat=np.zeros([2,2])
	age_confmat=np.zeros([14,14])
	for ij, item in enumerate(test_loader):
		out=vggface_model(input=item[0])
		gender_out=gender_model.predict(out[0])
		age_out=age_model.predict(out[0])
		batch_confmat+=confusion_matrix(item[1], tf.argmax(gender_out, axis=1).numpy(), labels=range(2))
		age_confmat+=confusion_matrix(item[3], tf.argmax(age_out, axis=1).numpy(), labels=range(14))
		print('Testing iteration: %d, confmat: %s, male_acc: %f, female_acc: %f, age_acc:%f'%(ij, batch_confmat, batch_confmat[0,0]/np.sum(batch_confmat[0,:]), batch_confmat[1,1]/np.sum(batch_confmat[1,:]), np.sum(np.diag(age_confmat))/np.sum(age_confmat)))

if __name__ == "__main__":
	batch_size=128
	train_loader=tf.data.Dataset.from_generator(lambda: data_generator('train'), output_types=(np.float32, np.int32, np.int32, np.int32)).batch(batch_size)
	valid_loader=tf.data.Dataset.from_generator(lambda: data_generator('valid'), output_types=(np.float32, np.int32, np.int32, np.int32)).batch(batch_size)
	test_loader=tf.data.Dataset.from_generator(lambda: data_generator('test'), output_types=(np.float32, np.int32, np.int32, np.int32)).batch(batch_size)
	# sess=onnxruntime.InferenceSession('/home/sriram/Documents/gender_challenge/vgg_face_torch/vggface.onnx', None)
	# input_name, output_name= sess.get_inputs()[0].name, sess.get_outputs()[0].name
	vggface_model=tf.saved_model.load('vggface_tensorflow.pb')
	gender_model=gender_net()
	age_model=age_net()
	gender_op=tf.keras.optimizers.Adam(1e-5)
	age_op=tf.keras.optimizers.Adam(1e-5)
	criterion= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	gender_model.compile(optimizer=gender_op, loss=criterion)
	age_model.compile(optimizer=age_op, loss=criterion)
				
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	prefix='gendernet_%s_agenet_%s'%(str([1500,750,250,2]), str([2000,1250,500,14]))
	writer=tf.summary.create_file_writer('runs/%s_%s'%(current_time, prefix))
	# train(train_loader, valid_loader, vggface_model, gender_model, gender_op, criterion, writer, age_model)
	test_model(test_loader, vggface_model, gender_model, age_model)