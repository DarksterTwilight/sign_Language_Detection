  *	?K7?A?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?~? ???!1?e???L@)??????1??"???K@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??u?????!^??,#?@)c`???1???֕>@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?)?T???!?#o??iC@)???z?2??1e?????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate????#??!-h??!??)???T??1??1????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat/?????!?8?ZM??)D??<????1# 2xF??:Preprocessing2U
Iterator::Model::ParallelMapV2?̕A???!X? g???)?̕A???1X? g???:Preprocessing2F
Iterator::Model??S9?)??!?S?R]???)?7?ܘ???1???SH??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate[0]::TensorSlice?h?󊧎?!GZ;֕???)?h?󊧎?1GZ;֕???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeZ?N????!M?4????)Z?N????1M?4????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipZ?Pۆ??!uQ???M@)???"Ɔ?1??}????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorҩ+??y??!??߈???)ҩ+??y??1??߈???:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch'???S~?!:N??e???)'???S~?1:N??e???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate?=yX???!?(?Wn??)h?.?KRy?129?????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[1]::FromTensor8????CI?!???՞?)8????CI?1???՞?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.