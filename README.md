# mygitdemo
https://www.lizenghai.com/archives/66992.html

5、推理
5.1 模型导出
在训练过程中，会保存模型文件到硬盘，如下：


其形式是TensorFlow的checkpoint格式，代码中提供了一个脚本（export_model.py）可以将checkpoint转换为.pb格式。

export_model.py主要参数：

checkpoint_path：训练保存的检查点文件
export_path：模型导出路径
num_classes：分类类别
crop_size：图像尺寸，[513, 513]
atrous_rates：12, 24, 36
output_stride：8
生成的.pb文件如下：

在这里插入图片描述

5.2 单张图像上推理
class DeepLabModel(object):
    """class to load deeplab model and run inference"""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME='SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME= 'frozen_inference_graph'
    def __init__(self, pretrained_weights):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None
        # Extract frozen graph from tar archive
        if pretrained_weights.endswith('.tar.gz'):
            tar_file = tarfile.open(pretrained_weights)
            for tar_info in tar_file.getmembers():
                if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                    file_handle = tar_file.extractfile(tar_info)
                    graph_def = tf.GraphDef.FromString(file_handle.read())
                    break
            tar_file.close()
        else:
            with open(pretrained_weights, 'rb') as fd:
                graph_def = tf.GraphDef.FromString(fd.read())
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=config)
    def run(self, image):
        """Runs inference on a single image.
        Args:
            image: A PIL.Image object, raw input image.
        Returns:
            resized_image:RGB image resized from original input image.
            seg_map:Segmentation map of 'resized_iamge'.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE/max(width, height)
        target_size = (int(resize_ratio*width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME:[np.asarray(resized_image)]}
        )
        seg_map = batch_seg_map[0]
        return resized_image, seg_map
        
        
if __name__ == '__main__':
    pretrained_weights = './train_logs/frozen_inference_graph_20000.pb'
    MODEL = DeepLabModel(pretrained_weights) # 加载模型
    
    img_name = 'test.jpg'
    img = Image.open(img_name)
    resized_im, seg_map = MODEL.run(original_im) #获取结果
    seg_map[seg_map==1]=255 #将人像的像素值置为255
    seg_map.save('output.jpg') # 保存mask结果图像
