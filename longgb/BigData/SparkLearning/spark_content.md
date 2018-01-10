
公共类 Public classes:

SparkConf: For configuring Spark.
SparkContext: Spark 的主入口.
SparkFiles: Access files shipped with jobs.
RDD: 一个 Resilient Distributed Dataset (RDD) 弹性分布式数据集.
Broadcast: A broadcast variable that gets reused across tasks.
Accumulator: An “add-only” shared variable that tasks can only add values to.
StorageLevel: Finer-grained cache persistence levels.

## 一、SparkConf 类
class pyspark.SparkConf(loadDefaults=True, _jvm=None, _jconf=None)
基本配置. 用于将各种Spark参数设置为键值对。 注意一旦SparkConf对象被传递给Spark，它就被克隆，不能再被用户修改。

方法：
>> setAppName(value)		设置应用名 Set application name.
>> setMaster(value)			Set master URL to connect to.
contains(key)				Does this configuration contain a given key?
get(key, defaultValue=None)		Get the configured value for some key, or return a default otherwise.
getAll()					Get all values as a list of key-value pairs.
set(key, value)				Set a configuration property.
setAll(pairs)				Set multiple parameters, passed as a list of key-value pairs.
setExecutorEnv(key=None, value=None, pairs=None) 	Set an environment variable to be passed to executors.
setIfMissing(key, value)	Set a configuration property, if not already set.
setSparkHome(value)			Set path where Spark is installed on worker nodes.
toDebugString()				Returns a printable version of the configuration, as a list of key=value pairs, one per line.


## 二、SparkContext 类
class pyspark.SparkContext(master=None, appName=None, sparkHome=None, pyFiles=None, environment=None, batchSize=0, 
							serializer=PickleSerializer(), conf=None, gateway=None, jsc=None, profiler_cls=<class 'pyspark.profiler.BasicProfiler'>)
Spark 的主入口，代表对集群的连接。SparkContext表示与Spark集群的连接，可用于在该集群上创建RDD和广播变量。
PACKAGE_EXTENSIONS = ('.zip', '.egg', '.jar')

方法：
>> accumulator(value, accum_param=None)	用给定的初始值创建一个累加器，使用给定的AccumulatorParam助手对象来定义如何提供数据类型的值。 默认AccumulatorParams用于整数和浮点数，如果你没有提供。 对于其他类型，可以使用自定义的AccumulatorParam。
>> applicationId						唯一的 identifier 在 Spark application. 
```
# spark 的applicationId是 ‘local-1433865536131’
# YARN 的是 ‘application_1433865536131_34483’
>>> sc.applicationId  
u'local-...'
```
>> broadcast(value)						在集群上广播一个只读变量在集群上, 返回 L{Broadcast<pyspark.broadcast.Broadcast>} 在分布式函数中读取它的对象. 
>> defaultMinPartitions					默认最小的 partitions 数目 for Hadoop RDDs 当不指定的时候
>> defaultParallelism					默认 parallelism 水平当使用的时候且用户不指定 (e.g. for reduce tasks)
>> emptyRDD()							创建 no partitions or elements 的 RDD 
>> getConf()							返回 pyspark.conf.SparkConf
>> classmethod getOrCreate(conf=None)	获取或实例化一个SparkContext并将其注册为一个单例对象。
		Parameters:	conf – SparkConf (optional)
>> cancelJobGroup(groupId)				取消指定组的活动作业。 有关更多信息，请参见SparkContext.setJobGroup。
>> setJobGroup(groupId, description, interruptOnCancel=False)	为此线程启动的所有作业分配一个组ID，直到组ID被设置为不同的值或清除。 通常，应用程序中的执行单元由多个Spark操作或作业组成。 应用程序员可以使用这种方法将所有这些作业分组在一起，并给出一个组描述。 一旦设置，Spark Web UI将把这些作业与这个组关联起来。 应用程序可以使用SparkContext.cancelJobGroup取消该组中的所有正在运行的作业。 取消如果作业组的interruptOnCancel设置为true，则作业取消将导致在作业的执行程序线程上调用Thread.interrupt（）。 这有助于确保任务实际上被及时停止，但由于HDFS-1208的原因，默认情况下会关闭，HDFS可能会通过将节点标记为死亡来响应Thread.interrupt（）。
```
>>> import threading
>>> from time import sleep
>>> result = "Not Set"
>>> lock = threading.Lock()
>>> def map_func(x):
...     sleep(100)
...     raise Exception("Task should have been cancelled")
>>> def start_job(x):
...     global result
...     try:
...         sc.setJobGroup("job_to_cancel", "some description")
...         result = sc.parallelize(range(x)).map(map_func).collect()
...     except Exception as e:
...         result = "Cancelled"
...     lock.release()
>>> def stop_job():
...     sleep(5)
...     sc.cancelJobGroup("job_to_cancel")
>>> supress = lock.acquire()
>>> supress = threading.Thread(target=start_job, args=(10,)).start()
>>> supress = threading.Thread(target=stop_job).start()
>>> supress = lock.acquire()
>>> print(result)
```
>> parallelize(c, numSlices=None)							把 local 的 Python collection 发散形成一个 RDD
```
>>> sc.parallelize([0, 2, 3, 4, 6], 5).glom().collect()
[[0], [2], [3], [4], [6]]
>>> sc.parallelize(xrange(0, 6, 2), 5).glom().collect()
[[], [0], [], [2], [4]]
```
>> pickleFile(name, minPartitions=None)						读取 RDD.saveAsPickleFile 存储的文件.
```
>>> tmpFile = NamedTemporaryFile(delete=True)
>>> tmpFile.close()
>>> sc.parallelize(range(10)).saveAsPickleFile(tmpFile.name, 5)
>>> sorted(sc.pickleFile(tmpFile.name, 3).collect())
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```
>> range(start, end=None, step=1, numSlices=None)			创建一个 RDD. sc.range()
		Parameters:	
				numSlices – the number of partitions of the new RDD
```
>>> sc.range(5).collect()
[0, 1, 2, 3, 4]
>>> sc.range(2, 4).collect()
[2, 3]
>>> sc.range(1, 7, 2).collect()
[1, 3, 5]
```
>> runJob(rdd, partitionFunc, partitions=None, allowLocal=False)			在指定的一组分区上执行给定的partitionFunc，并将结果作为元素数组返回。 如果 partitions 没有被指定，则运行在所有的分区上
```
>>> myRDD = sc.parallelize(range(6), 3)
>>> sc.runJob(myRDD, lambda part: [x * x for x in part])
[0, 1, 4, 9, 16, 25]
>>> myRDD = sc.parallelize(range(6), 3)
>>> sc.runJob(myRDD, lambda part: [x * x for x in part], [0, 2], True)
[0, 1, 16, 25]
```
>> textFile(name, minPartitions=None, use_unicode=True)						读取 HDFS 的 text file、 local file system (available on all nodes), or any Hadoop-supported file system URI, 返回 Strings 的 RDD.If use_unicode is False, the strings will be kept as str (encoding as utf-8), which is faster and smaller than unicode. (Added in Spark 1.2)
```
>>> path = os.path.join(tempdir, "sample-text.txt")
>>> with open(path, "w") as testFile:
...    _ = testFile.write("Hello world!")
>>> textFile = sc.textFile(path)
>>> textFile.collect()
[u'Hello world!']
uiWebUrl
```
>> union(rdds)										Build the union of a list of RDDs. This supports unions() of RDDs with different serialized formats, although this forces them to be reserialized using the default serializer:
```
>>> path = os.path.join(tempdir, "union-text.txt")
>>> with open(path, "w") as testFile:
...    _ = testFile.write("Hello")
>>> textFile = sc.textFile(path)
>>> textFile.collect()
[u'Hello']
>>> parallelized = sc.parallelize(["World!"])
>>> sorted(sc.union([textFile, parallelized]).collect())
[u'Hello', 'World!']
```
>> wholeTextFiles(path, minPartitions=None, use_unicode=True)			读取 HDFS 的 text files、 local file system (available on all nodes), or any Hadoop-supported file system URI. 每个文件是单个记录返回 key-value pair, key 是文件路径, value 文件内容. If use_unicode is False, the strings will be kept as str (encoding as utf-8), which is faster and smaller than unicode. (Added in Spark 1.2)
```
For example, if you have the following files:
hdfs://a-hdfs-path/part-00000
hdfs://a-hdfs-path/part-00001
...
hdfs://a-hdfs-path/part-nnnnn
Do rdd = sparkContext.wholeTextFiles(“hdfs://a-hdfs-path”), then rdd contains:
(a-hdfs-path/part-00000, its content)
(a-hdfs-path/part-00001, its content)
...
(a-hdfs-path/part-nnnnn, its content)
Note Small files are preferred, as each file will be loaded fully in memory.
>>> dirPath = os.path.join(tempdir, "files")
>>> os.mkdir(dirPath)
>>> with open(os.path.join(dirPath, "1.txt"), "w") as file1:
...    _ = file1.write("1")
>>> with open(os.path.join(dirPath, "2.txt"), "w") as file2:
...    _ = file2.write("2")
>>> textFiles = sc.wholeTextFiles(dirPath)
>>> sorted(textFiles.collect())
[(u'.../1.txt', u'1'), (u'.../2.txt', u'2')]
```
addFile(path, recursive=False)			在每个节点上启动spark job来下载文件。路径可以是a local file、 HDFS、 HTTP、 HTTPS or FTP URI.在 Spark jobs 上连接文件, use L{SparkFiles.get(fileName)<pyspark.files.SparkFiles.get>} with the filename to find its download location.
		A directory can be given if the 递归选项是 True. Currently directories are only supported for Hadoop-supported filesystems.
```
>>> from pyspark import SparkFiles
>>> path = os.path.join(tempdir, "test.txt")
>>> with open(path, "w") as testFile:
...    _ = testFile.write("100")
>>> sc.addFile(path)
>>> def func(iterator):
...    with open(SparkFiles.get("test.txt")) as testFile:
...        fileVal = int(testFile.readline())
...        return [x * fileVal for x in iterator]
>>> sc.parallelize([1, 2, 3, 4]).mapPartitions(func).collect()
[100, 200, 300, 400]
```
addPyFile(path)							Add a .py or .zip dependency for all tasks to be executed on this SparkContext in the future. The path passed can be either a local file, a file in HDFS (or other Hadoop-supported filesystems), or an HTTP, HTTPS or FTP URI.
binaryFiles(path, minPartitions=None)	Note Experimental 	Read a directory of binary files from HDFS, a local file system (available on all nodes), or any Hadoop-supported file system URI as a byte array. Each file is read as a single record and returned in a key-value pair, where the key is the path of each file, the value is the content of each file. Note Small files are preferred, large file is also allowable, but may cause bad performance.
binaryRecords(path, recordLength)		Note Experimental	Load data from a flat binary file, assuming each record is a set of numbers with the specified numerical format (see ByteBuffer), and the number of bytes per record is constant.
		Parameters:	
				path – Directory to the input data files
				recordLength – The length at which to split the records
cancelAllJobs()							取消所有任务.
dump_profiles(path)						Dump the profile stats into directory path
getLocalProperty(key)					Get a local property set in this thread, or null if it is missing. See setLocalProperty
hadoopFile(path, inputFormatClass, keyClass, valueClass, keyConverter=None, valueConverter=None, conf=None, batchSize=0)		读取旧的 Hadoop InputFormat with arbitrary key and value class from HDFS, a local file system (available on all nodes), or any Hadoop-supported file system URI. The mechanism is the same as for sc.sequenceFile. A Hadoop configuration can be passed in as a Python dict. This will be converted into a Configuration in Java.
		Parameters:	
				path – path to Hadoop file
				inputFormatClass – fully qualified classname of Hadoop InputFormat (e.g. “org.apache.hadoop.mapred.TextInputFormat”)
				keyClass – fully qualified classname of key Writable class (e.g. “org.apache.hadoop.io.Text”)
				valueClass – fully qualified classname of value Writable class (e.g. “org.apache.hadoop.io.LongWritable”)
				keyConverter – (None by default)
				valueConverter – (None by default)
				conf – Hadoop configuration, passed in as a dict (None by default)
				batchSize – The number of Python objects represented as a single Java object. (default 0, choose batchSize automatically)
hadoopRDD(inputFormatClass, keyClass, valueClass, keyConverter=None, valueConverter=None, conf=None, batchSize=0)				读取旧的 Hadoop InputFormat with arbitrary key and value class, from an arbitrary Hadoop configuration, which is passed in as a Python dict. This will be converted into a Configuration in Java. The mechanism is the same as for sc.sequenceFile.
		Parameters:	
				inputFormatClass – fully qualified classname of Hadoop InputFormat (e.g. “org.apache.hadoop.mapred.TextInputFormat”)
				keyClass – fully qualified classname of key Writable class (e.g. “org.apache.hadoop.io.Text”)
				valueClass – fully qualified classname of value Writable class (e.g. “org.apache.hadoop.io.LongWritable”)
				keyConverter – (None by default)
				valueConverter – (None by default)
				conf – Hadoop configuration, passed in as a dict (None by default)
				batchSize – The number of Python objects represented as a single Java object. (default 0, choose batchSize automatically)
newAPIHadoopFile(path, inputFormatClass, keyClass, valueClass, keyConverter=None, valueConverter=None, conf=None, batchSize=0)	读取新的 API Hadoop InputFormat with arbitrary key and value class from HDFS, a local file system (available on all nodes), or any Hadoop-supported file system URI. The mechanism is the same as for sc.sequenceFile. A Hadoop configuration can be passed in as a Python dict. This will be converted into a Configuration in Java
		Parameters:	
				path – path to Hadoop file
				inputFormatClass – fully qualified classname of Hadoop InputFormat (e.g. “org.apache.hadoop.mapreduce.lib.input.TextInputFormat”)
				keyClass – fully qualified classname of key Writable class (e.g. “org.apache.hadoop.io.Text”)
				valueClass – fully qualified classname of value Writable class (e.g. “org.apache.hadoop.io.LongWritable”)
				keyConverter – (None by default)
				valueConverter – (None by default)
				conf – Hadoop configuration, passed in as a dict (None by default)
				batchSize – The number of Python objects represented as a single Java object. (default 0, choose batchSize automatically)
newAPIHadoopRDD(inputFormatClass, keyClass, valueClass, keyConverter=None, valueConverter=None, conf=None, batchSize=0)			读取新的API Hadoop InputFormat with arbitrary key and value class, from an arbitrary Hadoop configuration, which is passed in as a Python dict. This will be converted into a Configuration in Java. The mechanism is the same as for sc.sequenceFile.
		Parameters:	
				inputFormatClass – fully qualified classname of Hadoop InputFormat (e.g. “org.apache.hadoop.mapreduce.lib.input.TextInputFormat”)
				keyClass – fully qualified classname of key Writable class (e.g. “org.apache.hadoop.io.Text”)
				valueClass – fully qualified classname of value Writable class (e.g. “org.apache.hadoop.io.LongWritable”)
				keyConverter – (None by default)
				valueConverter – (None by default)
				conf – Hadoop configuration, passed in as a dict (None by default)
				batchSize – The number of Python objects represented as a single Java object. (default 0, choose batchSize automatically)

sequenceFile(path, keyClass=None, valueClass=None, keyConverter=None, valueConverter=None, minSplits=None, batchSize=0)			Read a Hadoop SequenceFile with arbitrary key and value Writable class from HDFS, a local file system (available on all nodes), or any Hadoop-supported file system URI. The mechanism is as follows:	A Java RDD is created from the SequenceFile or other InputFormat, and the key and value Writable classes 	Serialization is attempted via Pyrolite pickling	If this fails, the fallback is to call ‘toString’ on each key and value	PickleSerializer is used to deserialize pickled objects on the Python side
		Parameters:	
				path – path to sequncefile
				keyClass – fully qualified classname of key Writable class (e.g. “org.apache.hadoop.io.Text”)
				valueClass – fully qualified classname of value Writable class (e.g. “org.apache.hadoop.io.LongWritable”)
				keyConverter –
				valueConverter –
				minSplits – minimum splits in dataset (default min(2, sc.defaultParallelism))
				batchSize – The number of Python objects represented as a single Java object. (default 0, choose batchSize automatically)
setCheckpointDir(dirName)							Set the directory under which RDDs are going to be checkpointed. The directory must be a HDFS path if running on a cluster.
setLocalProperty(key, value)						Set a local property that affects jobs submitted from this thread, such as the Spark fair scheduler pool.
setLogLevel(logLevel)								Control our logLevel. This overrides any user-defined log settings. Valid log levels include: ALL, DEBUG, ERROR, FATAL, INFO, OFF, TRACE, WARN
classmethod setSystemProperty(key, value)			Set a Java system property, such as spark.executor.memory. This must must be invoked before instantiating SparkContext.
show_profiles()										Print the profile stats to stdout
sparkUser()											Get SPARK_USER for user who is running SparkContext.
startTime											Return the epoch time when the Spark Context was started.
statusTracker()										Return StatusTracker object
stop()												Shut down the SparkContext.
version												The version of Spark on which this application is running.

## 三、SparkFiles 类
class pyspark.SparkFiles
Resolves paths to files added through L{SparkContext.addFile()<pyspark.context.SparkContext.addFile>}.
SparkFiles contains only classmethods; users should not create SparkFiles instances.

classmethod get(filename)				Get the absolute path of a file added through SparkContext.addFile().
classmethod getRootDirectory()			Get the root directory that contains files added through SparkContext.addFile().

## 四、RDD 类
class pyspark.RDD(jrdd, ctx, jrdd_deserializer=AutoBatchedSerializer(PickleSerializer()))
A Resilient Distributed Dataset (RDD). 不可改变的, 分区能够被并行运行.

方法：
>> aggregate(zeroValue, seqOp, combOp)				聚合每一个分区, and then the results for all the partitions, using a given combine functions and a neutral “zero value.” The functions op(t1, t2) is allowed to modify t1 and return it as its result value to avoid object allocation; however, it should not modify t2. The first function (seqOp) can return a different result type, U, than the type of this RDD. Thus, we need one operation for merging a T into an U and one operation for merging two U
< "http://blog.csdn.net/qingyang0320/article/details/51603243" >
```
>>> seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
>>> combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
>>> sc.parallelize([1, 2, 3, 4]).aggregate((0, 0), seqOp, combOp)
(10, 4)
>>> sc.parallelize([]).aggregate((0, 0), seqOp, combOp)
(0, 0)
```
>> aggregateByKey(zeroValue, seqFunc, combFunc, numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)		Aggregate the values of each key, using given combine functions and a neutral “zero value”. This function can return a different result type, U, than the type of the values in this RDD, V. Thus, we need one operation for merging a V into a U and one operation for merging two U’s, The former operation is used for merging values within a partition, and the latter is used for merging values between partitions. To avoid memory allocation, both of these functions are allowed to modify and return their first argument instead of creating a new U.
>> cache()											存储 Persist this RDD with the default storage level (MEMORY_ONLY).
>> cartesian(other)									笛卡尔积, 两个 RDD 的所有元素形成的 (a, b) .
```
>>> rdd = sc.parallelize([1, 2])
>>> sorted(rdd.cartesian(rdd).collect())
[(1, 1), (1, 2), (2, 1), (2, 2)]
```
>> coalesce(numPartitions, shuffle=False)			返回一个 RDD 被 reduced 成 numPartitions 个 partitions.
```
>>> sc.parallelize([1, 2, 3, 4, 5], 3).glom().collect()
[[1], [2, 3], [4, 5]]
>>> sc.parallelize([1, 2, 3, 4, 5], 3).coalesce(1).glom().collect()
[[1, 2, 3, 4, 5]]
```
>> cogroup(other, numPartitions=None)				For each key k in self or other, return a resulting RDD that contains a tuple with the list of values for that key in self as well as other.
```
>>> x = sc.parallelize([("a", 1), ("b", 4)])
>>> y = sc.parallelize([("a", 2)])
>>> [(x, tuple(map(list, y))) for x, y in sorted(list(x.cogroup(y).collect()))]
[('a', ([1], [2])), ('b', ([4], []))]
```
>> collect()										返回包含 RDD 所有元素一个 list 。这种方法应该使用在结果数据集比较小的情况。因为所有的数据背会读取到 driver 的内存里。
>> collectAsMap()									返回 key-value 的 dict. 这种方法应该使用在结果数据集比较小的情况。因为所有的数据背会读取到 driver 的内存里。
```
>>> m = sc.parallelize([(1, 2), (3, 4)]).collectAsMap()
>>> m[1]
2
>>> m[3]
4
```
>> combineByKey(createCombiner, mergeValue, mergeCombiners, numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)
		< "http://blog.csdn.net/hit0803107/article/details/52808986" >
		Turns an RDD[(K, V)] into a result of type RDD[(K, C)], for a “combined type” C.
		createCombiner, which turns a V into a C (e.g., creates a one-element list)
		mergeValue, 如果出现新的key，使用createCombiner，否则使用mergeValue，merge a V into a C (e.g., adds it to the end of a list)
		mergeCombiners, reduce过程 to combine two C’s into a single one.
```
>>> x = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
>>> def add(a, b): return a + str(b)
>>> sorted(x.combineByKey(str, add, add).collect())
[('a', '11'), ('b', '1')]
```
>> count()											Return the number of elements in this RDD.
```
>>> sc.parallelize([2, 3, 4]).count()
3
```
checkpoint()										Mark this RDD for checkpointing. It will be saved to a file inside the checkpoint directory set with SparkContext.setCheckpointDir() and all references to its parent RDDs will be removed. This function must be called before any job has been executed on this RDD. It is strongly recommended that this RDD is persisted in memory, otherwise saving it on a file will require recomputation.
context 											The SparkContext that this RDD was created on.
countApprox(timeout, confidence=0.95)				Note Experimental	Approximate version of count() that returns a potentially incomplete result within a timeout, even if not all tasks have finished.
```
>>> rdd = sc.parallelize(range(1000), 10)
>>> rdd.countApprox(1000, 1.0)
1000
```
countApproxDistinct(relativeSD=0.05)				Note Experimental	Return approximate number of distinct elements in the RDD.	The algorithm used is based on streamlib’s implementation of “HyperLogLog in Practice: Algorithmic Engineering of a State of The Art Cardinality Estimation Algorithm”, available here.
		Parameters:	relativeSD – Relative accuracy. Smaller values create counters that require more space. It must be greater than 0.000017.
```		
>>> n = sc.parallelize(range(1000)).map(str).countApproxDistinct()
>>> 900 < n < 1100
True
>>> n = sc.parallelize([i % 20 for i in range(1000)]).countApproxDistinct()
>>> 16 < n < 24
True
```
>> countByKey()										Count the number of elements for each key, and return the result to the master as a dictionary.
```
>>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
>>> sorted(rdd.countByKey().items())
[('a', 2), ('b', 1)]
```
>> countByValue()									Return the count of each unique value in this RDD as a dictionary of (value, count) pairs.
```
>>> sorted(sc.parallelize([1, 2, 1, 2, 2], 2).countByValue().items())
[(1, 2), (2, 3)]
```
>> distinct(numPartitions=None)						返回去重元素的 new RDD .
```
>>> sorted(sc.parallelize([1, 1, 2, 3]).distinct().collect())
[1, 2, 3]
```
>> filter(f)										返回满足条件的 new RDD .
```
>>> rdd = sc.parallelize([1, 2, 3, 4, 5])
>>> rdd.filter(lambda x: x % 2 == 0).collect()
[2, 4]
```
>> first()											返回 RDD 的第一个元素.
```
>>> sc.parallelize([2, 3, 4]).first()
2
>>> sc.parallelize([]).first()
Traceback (most recent call last):
    ...
ValueError: RDD is empty
```
>> flatMap(f, preservesPartitioning=False)			先 map 这个结果，然后把结果拉平，全部放在一起.
```
>>> rdd = sc.parallelize([2, 3, 4])
>>> sorted(rdd.flatMap(lambda x: range(1, x)).collect())
[1, 1, 1, 2, 2, 3]
>>> sorted(rdd.flatMap(lambda x: [(x, x), (x, x)]).collect())
[(2, 2), (2, 2), (3, 3), (3, 3), (4, 4), (4, 4)]
```
>> flatMapValues(f)									Pass each value in the key-value pair RDD through a flatMap function without changing the keys; this also retains the original RDD’s partitioning.
```
>>> x = sc.parallelize([("a", ["x", "y", "z"]), ("b", ["p", "r"])])
>>> def f(x): return x
>>> x.flatMapValues(f).collect()
[('a', 'x'), ('a', 'y'), ('a', 'z'), ('b', 'p'), ('b', 'r')]
```
>> fold(zeroValue, op)								Aggregate the elements of each partition, and then the results for all the partitions, using a given associative function and a neutral “zero value.”The function op(t1, t2) is allowed to modify t1 and return it as its result value to avoid object allocation; however, it should not modify t2.This behaves somewhat differently from fold operations implemented for non-distributed collections in functional languages like Scala. This fold operation may be applied to partitions individually, and then fold those results into the final result, rather than apply the fold to each element sequentially in some defined ordering. For functions that are not commutative, the result may differ from that of a fold applied to a non-distributed collection.
```
>>> from operator import add
>>> sc.parallelize([1, 2, 3, 4, 5]).fold(0, add)
15
```
>> foldByKey(zeroValue, func, numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)	Merge the values for each key using an associative function “func” and a neutral “zeroValue” which may be added to the result an arbitrary number of times, and must not change the result (e.g., 0 for addition, or 1 for multiplication.).
```
>>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
>>> from operator import add
>>> sorted(rdd.foldByKey(0, add).collect())
[('a', 2), ('b', 1)]
```
>> foreach(f)										Applies a function to all elements of this RDD.
```
>>> def f(x): print(x)
>>> sc.parallelize([1, 2, 3, 4, 5]).foreach(f)
```
>> foreachPartition(f)								Applies a function to each partition of this RDD.
```
>>> def f(iterator):
...     for x in iterator:
...          print(x)
>>> sc.parallelize([1, 2, 3, 4, 5]).foreachPartition(f)
```
>> fullOuterJoin(other, numPartitions=None)			Perform a right outer join of self and other.
```
>>> x = sc.parallelize([("a", 1), ("b", 4)])
>>> y = sc.parallelize([("a", 2), ("c", 8)])
>>> sorted(x.fullOuterJoin(y).collect())
[('a', (1, 2)), ('b', (4, None)), ('c', (None, 8))]
```
getCheckpointFile()									Gets the name of the file to which this RDD was checkpointed Not defined if RDD is checkpointed locally.
>> getNumPartitions()								Returns the number of partitions in RDD
```
>>> rdd = sc.parallelize([1, 2, 3, 4], 2)
>>> rdd.getNumPartitions()
2
```
getStorageLevel()									Get the RDD’s current storage level.
```
>>> rdd1 = sc.parallelize([1,2])
>>> rdd1.getStorageLevel()
StorageLevel(False, False, False, False, 1)
>>> print(rdd1.getStorageLevel())
Serialized 1x Replicated
```
>> glom()											Return an RDD created by coalescing all elements within each partition into a list.
```
>>> rdd = sc.parallelize([1, 2, 3, 4], 2)
>>> sorted(rdd.glom().collect())
[[1, 2], [3, 4]]
```
>> groupBy(f, numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)			Return an RDD of grouped items.
```
>>> rdd = sc.parallelize([1, 1, 2, 3, 5, 8])
>>> result = rdd.groupBy(lambda x: x % 2).collect()
>>> sorted([(x, sorted(y)) for (x, y) in result])
[(0, [2, 8]), (1, [1, 1, 3, 5])]
```
>> groupByKey(numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)
```
>>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
>>> sorted(rdd.groupByKey().mapValues(len).collect())
[('a', 2), ('b', 1)]
>>> sorted(rdd.groupByKey().mapValues(list).collect())
[('a', [1, 1]), ('b', [1])]
```
groupWith(other, *others)							Alias for cogroup but with support for multiple RDDs.
```
>>> w = sc.parallelize([("a", 5), ("b", 6)])
>>> x = sc.parallelize([("a", 1), ("b", 4)])
>>> y = sc.parallelize([("a", 2)])
>>> z = sc.parallelize([("b", 42)])
>>> [(x, tuple(map(list, y))) for x, y in sorted(list(w.groupWith(x, y, z).collect()))]
[('a', ([5], [1], [2], [])), ('b', ([6], [4], [], [42]))]
```
histogram(buckets)
```
>>> rdd = sc.parallelize(range(51))
>>> rdd.histogram(2)
([0, 25, 50], [25, 26])
>>> rdd.histogram([0, 5, 25, 50])
([0, 5, 25, 50], [5, 20, 26])
>>> rdd.histogram([0, 15, 30, 45, 60])  # evenly spaced buckets
([0, 15, 30, 45, 60], [15, 15, 15, 6])
>>> rdd = sc.parallelize(["ab", "ac", "b", "bd", "ef"])
>>> rdd.histogram(("a", "b", "c"))
(('a', 'b', 'c'), [2, 2])
```
id()												A unique ID for this RDD (within its SparkContext).
>> intersection(other)								两个RDC的交集，会把自己的重复的也剔除
```
# Note This method performs a shuffle internally.
>>> rdd1 = sc.parallelize([1, 10, 2, 3, 4, 5])
>>> rdd2 = sc.parallelize([1, 6, 2, 3, 7, 8])
>>> rdd1.intersection(rdd2).collect()
[1, 2, 3]
```
isCheckpointed()									Return whether this RDD is checkpointed and materialized, either reliably or locally.
>> isEmpty()										判断是否为空
```
# Note an RDD may be empty even when it has at least 1 partition.
>>> sc.parallelize([]).isEmpty()
True
>>> sc.parallelize([1]).isEmpty()
False
```
isLocallyCheckpointed()								Return whether this RDD is marked for local checkpointing.
>> join(other, numPartitions=None)					以 key 相交
```
>>> x = sc.parallelize([("a", 1), ("b", 4)])
>>> y = sc.parallelize([("a", 2), ("a", 3)])
>>> sorted(x.join(y).collect())
[('a', (1, 2)), ('a', (1, 3))]
```
>> keyBy(f)											为每个值创建 key
```
>>> x = sc.parallelize(range(0,3)).keyBy(lambda x: x*x)
>>> y = sc.parallelize(zip(range(0,5), range(0,5)))
>>> [(m, list(map(list, n))) for m, n in sorted(x.cogroup(y).collect())]
[(0, [[0], [0]]), (1, [[1], [1]]), (2, [[], [2]]), (3, [[], [3]]), (4, [[2], [4]])]
```
>> keys()											返回 key 值
```
>>> m = sc.parallelize([(1, 2), (3, 4)]).keys()
>>> m.collect()
[1, 3]
```
>> leftOuterJoin(other, numPartitions=None)			相当于 left outer join .
```
>>> x = sc.parallelize([("a", 1), ("b", 4)])
>>> y = sc.parallelize([("a", 2)])
>>> sorted(x.leftOuterJoin(y).collect())
[('a', (1, 2)), ('b', (4, None))]
```
localCheckpoint()									Mark this RDD for local checkpointing using Spark’s existing caching layer.This method is for users who wish to truncate RDD lineages while skipping the expensive step of replicating the materialized data in a reliable distributed file system. This is useful for RDDs with long lineages that need to be truncated periodically (e.g. GraphX).Local checkpointing sacrifices fault-tolerance for performance. In particular, checkpointed data is written to ephemeral local storage in the executors instead of to a reliable, fault-tolerant storage. The effect is that if an executor fails during the computation, the checkpointed data may no longer be accessible, causing an irrecoverable job failure.This is NOT safe to use with dynamic allocation, which removes executors along with their cached blocks. If you must use both features, you are advised to set spark.dynamicAllocation.cachedExecutorIdleTimeout to a high value.The checkpoint directory set through SparkContext.setCheckpointDir() is not used.
>> lookup(key)										返回查找的 key 的值, This operation is done efficiently if the RDD has a known partitioner by only searching the partition that the key maps to.
```
>>> l = range(1000)
>>> rdd = sc.parallelize(zip(l, l), 10)
>>> rdd.lookup(42)  # slow
[42]
>>> sorted = rdd.sortByKey()
>>> sorted.lookup(42)  # fast
[42]
>>> sorted.lookup(1024)
[]
>>> rdd2 = sc.parallelize([(('a', 'b'), 'c')]).groupByKey()
>>> list(rdd2.lookup(('a', 'b'))[0])
['c']
```
>> map(f, preservesPartitioning=False)				对 rdd 每个元素 map
```
>>> rdd = sc.parallelize(["b", "a", "c"])
>>> sorted(rdd.map(lambda x: (x, 1)).collect())
[('a', 1), ('b', 1), ('c', 1)]
```
>> mapPartitions(f, preservesPartitioning=False)	对 rdd 每个分区进行 map
```
>>> rdd = sc.parallelize([1, 2, 3, 4], 2)
>>> def f(iterator): yield sum(iterator)
>>> rdd.mapPartitions(f).collect()
[3, 7]
```
mapPartitionsWithIndex(f, preservesPartitioning=False)				Return a new RDD by applying a function to each partition of this RDD, while tracking the index of the original partition.
```
>>> rdd = sc.parallelize([1, 2, 3, 4], 4)
>>> def f(splitIndex, iterator): yield splitIndex
>>> rdd.mapPartitionsWithIndex(f).sum()
6
```
mapPartitionsWithSplit(f, preservesPartitioning=False)				Deprecated: use mapPartitionsWithIndex instead.Return a new RDD by applying a function to each partition of this RDD, while tracking the index of the original partition.
```
>>> rdd = sc.parallelize([1, 2, 3, 4], 4)
>>> def f(splitIndex, iterator): yield splitIndex
>>> rdd.mapPartitionsWithSplit(f).sum()
6
```
>> mapValues(f)										对 values 进行 map
```
>>> x = sc.parallelize([("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])])
>>> def f(x): return len(x)
>>> x.mapValues(f).collect()
[('a', 3), ('b', 1)]
```
>> max(key=None)									Parameters:	key – A function used to generate key for comparing
```
>>> rdd = sc.parallelize([1.0, 5.0, 43.0, 10.0])
>>> rdd.max()
43.0
>>> rdd.max(key=str)
5.0
```
>> mean()
```
>>> sc.parallelize([1, 2, 3]).mean()
2.0
```
meanApprox(timeout, confidence=0.95)				Note Experimental	Approximate operation to return the mean within a timeout or meet the confidence.
```
>>> rdd = sc.parallelize(range(1000), 10)
>>> r = sum(range(1000)) / 1000.0
>>> abs(rdd.meanApprox(1000) - r) / r < 0.05
True
```
>> min(key=None)									Parameters:	key – A function used to generate key for comparing
```
>>> rdd = sc.parallelize([2.0, 5.0, 43.0, 10.0])
>>> rdd.min()
2.0
>>> rdd.min(key=str)
10.0
```
name()												Return the name of this RDD.
>> partitionBy(numPartitions, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)				返回一个 copy of the RDD 按照指定分区
```
>>> pairs = sc.parallelize([1, 2, 3, 4, 2, 4, 1]).map(lambda x: (x, x))
>>> sets = pairs.partitionBy(2).glom().collect()
>>> len(set(sets[0]).intersection(set(sets[1])))
0
```
>> persist(storageLevel=StorageLevel(False, True, False, False, 1))									Set this RDD’s storage level to persist its values across operations after the first time it is computed. This can only be used to assign a new storage level if the RDD does not have a storage level set yet. If no storage level is specified defaults to (MEMORY_ONLY).
```
>>> rdd = sc.parallelize(["b", "a", "c"])
>>> rdd.persist().is_cached
True
```
pipe(command, env=None, checkCode=False)			Return an RDD created by piping elements to a forked external process. Parameters:	checkCode – whether or not to check the return value of the shell command.
```
>>> sc.parallelize(['1', '2', '', '3']).pipe('cat').collect()
[u'1', u'2', u'', u'3']
```
randomSplit(weights, seed=None)						Randomly splits this RDD with the provided weights.
```
>>> rdd = sc.parallelize(range(500), 1)
>>> rdd1, rdd2 = rdd.randomSplit([2, 3], 17)
>>> len(rdd1.collect() + rdd2.collect())
500
>>> 150 < rdd1.count() < 250
True
>>> 250 < rdd2.count() < 350
True
```
>> reduce(f)										Reduces the elements of this RDD using the specified commutative and associative binary operator. Currently reduces partitions locally.
```
>>> from operator import add
>>> sc.parallelize([1, 2, 3, 4, 5]).reduce(add)
15
>>> sc.parallelize((2 for _ in range(10))).map(lambda x: 1).cache().reduce(add)
10
>>> sc.parallelize([]).reduce(add)
Traceback (most recent call last):
    ...
ValueError: Can not reduce() empty RDD
```
>> reduceByKey(func, numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)				按照 key 进行分别 reduce
```
>>> from operator import add
>>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
>>> sorted(rdd.reduceByKey(add).collect())
[('a', 2), ('b', 1)]
```
reduceByKeyLocally(func)							Merge the values for each key using an associative and commutative reduce function, but return the results immediately to the master as a dictionary.	This will also perform the merging locally on each mapper before sending results to a reducer, similarly to a “combiner” in MapReduce.
```
>>> from operator import add
>>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
>>> sorted(rdd.reduceByKeyLocally(add).items())
[('a', 2), ('b', 1)]
```
repartition(numPartitions)							Return a new RDD that has exactly numPartitions partitions.	Can increase or decrease the level of parallelism in this RDD. Internally, this uses a shuffle to redistribute data. If you are decreasing the number of partitions in this RDD, consider using coalesce, which can avoid performing a shuffle.
```
>>> rdd = sc.parallelize([1,2,3,4,5,6,7], 4)
>>> sorted(rdd.glom().collect())
[[1], [2, 3], [4, 5], [6, 7]]
>>> len(rdd.repartition(2).glom().collect())
2
>>> len(rdd.repartition(10).glom().collect())
10
```
repartitionAndSortWithinPartitions(numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>, ascending=True, keyfunc=<function <lambda> at 0x7fc35dbcf758>)				Repartition the RDD according to the given partitioner and, within each resulting partition, sort records by their keys.
```
>>> rdd = sc.parallelize([(0, 5), (3, 8), (2, 6), (0, 8), (3, 8), (1, 3)])
>>> rdd2 = rdd.repartitionAndSortWithinPartitions(2, lambda x: x % 2, 2)
>>> rdd2.glom().collect()
[[(0, 5), (0, 8), (2, 6)], [(1, 3), (3, 8), (3, 8)]]
```
rightOuterJoin(other, numPartitions=None)			right outer join of self and other.
```
>>> x = sc.parallelize([("a", 1), ("b", 4)])
>>> y = sc.parallelize([("a", 2)])
>>> sorted(y.rightOuterJoin(x).collect())
[('a', (2, 1)), ('b', (None, 4))]
```
sample(withReplacement, fraction, seed=None)		抽样
		Parameters:	
				withReplacement – can elements be sampled multiple times (replaced when sampled out)
				fraction – expected size of the sample as a fraction of this RDD’s size without replacement: probability that each element is chosen; fraction must be [0, 1] with replacement: expected number of times each element is chosen; fraction must be >= 0
				seed – seed for the random number generator
				Note This is not guaranteed to provide exactly the fraction specified of the total count of the given DataFrame.
```
>>> rdd = sc.parallelize(range(100), 4)
>>> 6 <= rdd.sample(False, 0.1, 81).count() <= 14
True
```
sampleByKey(withReplacement, fractions, seed=None)	Return a subset of this RDD sampled by key (via stratified sampling). Create a sample of this RDD using variable sampling rates for different keys as specified by fractions, a key to sampling rate map.
```
>>> fractions = {"a": 0.2, "b": 0.1}
>>> rdd = sc.parallelize(fractions.keys()).cartesian(sc.parallelize(range(0, 1000)))
>>> sample = dict(rdd.sampleByKey(False, fractions, 2).groupByKey().collect())
>>> 100 < len(sample["a"]) < 300 and 50 < len(sample["b"]) < 150
True
>>> max(sample["a"]) <= 999 and min(sample["a"]) >= 0
True
>>> max(sample["b"]) <= 999 and min(sample["b"]) >= 0
True
```
sampleStdev()										样本标准差 Compute the sample standard deviation of this RDD’s elements (which corrects for bias in estimating the standard deviation by dividing by N-1 instead of N).
```
>>> sc.parallelize([1, 2, 3]).sampleStdev()
1.0
```
sampleVariance()									样本方差 Compute the sample variance of this RDD’s elements (which corrects for bias in estimating the variance by dividing by N-1 instead of N).
```
>>> sc.parallelize([1, 2, 3]).sampleVariance()
1.0
```
saveAsHadoopDataset(conf, keyConverter=None, valueConverter=None)					Output a Python RDD of key-value pairs (of form RDD[(K, V)]) to any Hadoop file system, using the old Hadoop OutputFormat API (mapred package). Keys/values are converted for output using either user specified converters or, by default, org.apache.spark.api.python.JavaToWritableConverter.
		Parameters:	
				conf – Hadoop job configuration, passed in as a dict
				keyConverter – (None by default)
				valueConverter – (None by default)
saveAsHadoopFile(path, outputFormatClass, keyClass=None, valueClass=None, keyConverter=None, valueConverter=None, conf=None, compressionCodecClass=None)			Output a Python RDD of key-value pairs (of form RDD[(K, V)]) to any Hadoop file system, using the old Hadoop OutputFormat API (mapred package). Key and value types will be inferred if not specified. Keys and values are converted for output using either user specified converters or org.apache.spark.api.python.JavaToWritableConverter. The conf is applied on top of the base Hadoop conf associated with the SparkContext of this RDD to create a merged Hadoop MapReduce job configuration for saving the data.
		Parameters:	
				path – path to Hadoop file
				outputFormatClass – fully qualified classname of Hadoop OutputFormat (e.g. “org.apache.hadoop.mapred.SequenceFileOutputFormat”)
				keyClass – fully qualified classname of key Writable class (e.g. “org.apache.hadoop.io.IntWritable”, None by default)
				valueClass – fully qualified classname of value Writable class (e.g. “org.apache.hadoop.io.Text”, None by default)
				keyConverter – (None by default)
				valueConverter – (None by default)
				conf – (None by default)
				compressionCodecClass – (None by default)
saveAsNewAPIHadoopDataset(conf, keyConverter=None, valueConverter=None)				Output a Python RDD of key-value pairs (of form RDD[(K, V)]) to any Hadoop file system, using the new Hadoop OutputFormat API (mapreduce package). Keys/values are converted for output using either user specified converters or, by default, org.apache.spark.api.python.JavaToWritableConverter.
		Parameters:	
				conf – Hadoop job configuration, passed in as a dict
				keyConverter – (None by default)
				valueConverter – (None by default)
saveAsNewAPIHadoopFile(path, outputFormatClass, keyClass=None, valueClass=None, keyConverter=None, valueConverter=None, conf=None)									Output a Python RDD of key-value pairs (of form RDD[(K, V)]) to any Hadoop file system, using the new Hadoop OutputFormat API (mapreduce package). Key and value types will be inferred if not specified. Keys and values are converted for output using either user specified converters or org.apache.spark.api.python.JavaToWritableConverter. The conf is applied on top of the base Hadoop conf associated with the SparkContext of this RDD to create a merged Hadoop MapReduce job configuration for saving the data.
		Parameters:	
				path – path to Hadoop file
				outputFormatClass – fully qualified classname of Hadoop OutputFormat (e.g. “org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat”)
				keyClass – fully qualified classname of key Writable class (e.g. “org.apache.hadoop.io.IntWritable”, None by default)
				valueClass – fully qualified classname of value Writable class (e.g. “org.apache.hadoop.io.Text”, None by default)
				keyConverter – (None by default)
				valueConverter – (None by default)
				conf – Hadoop job configuration, passed in as a dict (None by default)
saveAsPickleFile(path, batchSize=10)				Save this RDD as a SequenceFile of serialized objects. The serializer used is pyspark.serializers.PickleSerializer, default batch size is 10.
```
>>> tmpFile = NamedTemporaryFile(delete=True)
>>> tmpFile.close()
>>> sc.parallelize([1, 2, 'spark', 'rdd']).saveAsPickleFile(tmpFile.name, 3)
>>> sorted(sc.pickleFile(tmpFile.name, 5).map(str).collect())
['1', '2', 'rdd', 'spark']
```
saveAsSequenceFile(path, compressionCodecClass=None)								Output a Python RDD of key-value pairs (of form RDD[(K, V)]) to any Hadoop file system, using the org.apache.hadoop.io.Writable types that we convert from the RDD’s key and value types. The mechanism is as follows:	Pyrolite is used to convert pickled Python RDD into RDD of Java objects.	Keys and values of this Java RDD are converted to Writables and written out.
		Parameters:	
				path – path to sequence file
				compressionCodecClass – (None by default)
>> saveAsTextFile(path, compressionCodecClass=None)									Save this RDD as a text file, using string representations of elements.
		Parameters:	
				path – path to text file
				compressionCodecClass – (None by default) string i.e. “org.apache.hadoop.io.compress.GzipCodec”
```
>>> tempFile = NamedTemporaryFile(delete=True)
>>> tempFile.close()
>>> sc.parallelize(range(10)).saveAsTextFile(tempFile.name)
>>> from fileinput import input
>>> from glob import glob
>>> ''.join(sorted(input(glob(tempFile.name + "/part-0000*"))))
'0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n'
>>> tempFile2 = NamedTemporaryFile(delete=True)
>>> tempFile2.close()
>>> sc.parallelize(['', 'foo', '', 'bar', '']).saveAsTextFile(tempFile2.name)
>>> ''.join(sorted(input(glob(tempFile2.name + "/part-0000*"))))
'\n\n\nbar\nfoo\n'
# Using compressionCodecClass
>>> tempFile3 = NamedTemporaryFile(delete=True)
>>> tempFile3.close()
>>> codec = "org.apache.hadoop.io.compress.GzipCodec"
>>> sc.parallelize(['foo', 'bar']).saveAsTextFile(tempFile3.name, codec)
>>> from fileinput import input, hook_compressed
>>> result = sorted(input(glob(tempFile3.name + "/part*.gz"), openhook=hook_compressed))
>>> b''.join(result).decode('utf-8')
u'bar\nfoo\n'
```
setName(name)									Assign a name to this RDD.
```
>>> rdd1 = sc.parallelize([1, 2])
>>> rdd1.setName('RDD1').name()
u'RDD1'
```
>> sortBy(keyfunc, ascending=True, numPartitions=None)									排序
```
>>> tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
>>> sc.parallelize(tmp).sortBy(lambda x: x[0]).collect()
[('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
>>> sc.parallelize(tmp).sortBy(lambda x: x[1]).collect()
[('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
```
>> sortByKey(ascending=True, numPartitions=None, keyfunc=<function <lambda> at 0x7fc35dbcf848>)						按照 key 排序
```
>>> tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
>>> sc.parallelize(tmp).sortByKey().first()
('1', 3)
>>> sc.parallelize(tmp).sortByKey(True, 1).collect()
[('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
>>> sc.parallelize(tmp).sortByKey(True, 2).collect()
[('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
>>> tmp2 = [('Mary', 1), ('had', 2), ('a', 3), ('little', 4), ('lamb', 5)]
>>> tmp2.extend([('whose', 6), ('fleece', 7), ('was', 8), ('white', 9)])
>>> sc.parallelize(tmp2).sortByKey(True, 3, keyfunc=lambda k: k.lower()).collect()
[('a', 3), ('fleece', 7), ('had', 2), ('lamb', 5),...('white', 9), ('whose', 6)]
```
stats()											Return a StatCounter object that captures the mean, variance and count of the RDD’s elements in one operation.
stdev()											Compute the standard deviation of this RDD’s elements.
```
>>> sc.parallelize([1, 2, 3]).stdev()
0.816...
```
>> subtract(other, numPartitions=None)			差集
```
>>> x = sc.parallelize([("a", 1), ("b", 4), ("b", 5), ("a", 3)])
>>> y = sc.parallelize([("a", 3), ("c", None)])
>>> sorted(x.subtract(y).collect())
[('a', 1), ('b', 4), ('b', 5)]
```
>> subtractByKey(other, numPartitions=None)		差集，按照 key 来看，不看 value
```
>>> x = sc.parallelize([("a", 1), ("b", 4), ("b", 5), ("a", 2)])
>>> y = sc.parallelize([("a", 3), ("c", None)])
>>> sorted(x.subtractByKey(y).collect())
[('b', 4), ('b', 5)]
```
>> sum()										Add up the elements in this RDD.
```
>>> sc.parallelize([1.0, 2.0, 3.0]).sum()
6.0
```
sumApprox(timeout, confidence=0.95)				Note Experimental	Approximate operation to return the sum within a timeout or meet the confidence.
```
>>> rdd = sc.parallelize(range(1000), 10)
>>> r = sum(range(1000))
>>> abs(rdd.sumApprox(1000) - r) / r < 0.05
True
```
>> take(num)									取前 num 个元素. It works by first scanning one partition, and use the results from that partition to estimate the number of additional partitions needed to satisfy the limit.	Translated from the Scala implementation in RDD#take().
```
# Note this method should only be used if the resulting array is expected to be small, as all the data is loaded into the driver’s memory.
>>> sc.parallelize([2, 3, 4, 5, 6]).cache().take(2)
[2, 3]
>>> sc.parallelize([2, 3, 4, 5, 6]).take(10)
[2, 3, 4, 5, 6]
>>> sc.parallelize(range(100), 100).filter(lambda x: x > 90).take(3)
[91, 92, 93]
```
takeOrdered(num, key=None)						Get the N elements from an RDD ordered in ascending order or as specified by the optional key function.
```
>>> sc.parallelize([10, 1, 2, 9, 3, 4, 5, 6, 7]).takeOrdered(6)
[1, 2, 3, 4, 5, 6]
>>> sc.parallelize([10, 1, 2, 9, 3, 4, 5, 6, 7], 2).takeOrdered(6, key=lambda x: -x)
[10, 9, 7, 6, 5, 4]
```
takeSample(withReplacement, num, seed=None)		Return a fixed-size sampled subset of this RDD.
```
>>> rdd = sc.parallelize(range(0, 10))
>>> len(rdd.takeSample(True, 20, 1))
20
>>> len(rdd.takeSample(False, 5, 2))
5
>>> len(rdd.takeSample(False, 15, 3))
10
```
toDebugString()									A description of this RDD and its recursive dependencies for debugging.
toLocalIterator()								Return an iterator that contains all of the elements in this RDD. The iterator will consume as much memory as the largest partition in this RDD.
```
>>> rdd = sc.parallelize(range(10))
>>> [x for x in rdd.toLocalIterator()]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```
>> top(num, key=None)							Get the top N elements from an RDD.
```
>>> sc.parallelize([10, 4, 2, 12, 3]).top(1)
[12]
>>> sc.parallelize([2, 3, 4, 5, 6], 2).top(2)
[6, 5]
>>> sc.parallelize([10, 4, 2, 12, 3]).top(3, key=str)
[4, 3, 2]
```
treeAggregate(zeroValue, seqOp, combOp, depth=2)						Aggregates the elements of this RDD in a multi-level tree pattern.Parameters:	depth – suggested depth of the tree (default: 2)
```
>>> add = lambda x, y: x + y
>>> rdd = sc.parallelize([-5, -4, -3, -2, -1, 1, 2, 3, 4], 10)
>>> rdd.treeAggregate(0, add, add)
-5
>>> rdd.treeAggregate(0, add, add, 1)
-5
>>> rdd.treeAggregate(0, add, add, 2)
-5
>>> rdd.treeAggregate(0, add, add, 5)
-5
>>> rdd.treeAggregate(0, add, add, 10)
-5
```
treeReduce(f, depth=2)							Reduces the elements of this RDD in a multi-level tree pattern.Parameters:	depth – suggested depth of the tree (default: 2)
```
>>> add = lambda x, y: x + y
>>> rdd = sc.parallelize([-5, -4, -3, -2, -1, 1, 2, 3, 4], 10)
>>> rdd.treeReduce(add)
-5
>>> rdd.treeReduce(add, 1)
-5
>>> rdd.treeReduce(add, 2)
-5
>>> rdd.treeReduce(add, 5)
-5
>>> rdd.treeReduce(add, 10)
-5
```
>> union(other)									Return the union of this RDD and another one.
```
>>> rdd = sc.parallelize([1, 1, 2, 3])
>>> rdd.union(rdd).collect()
[1, 1, 2, 3, 1, 1, 2, 3]
```
unpersist()										Mark the RDD as non-persistent, and remove all blocks for it from memory and disk.
>> values()										Return an RDD with the values of each tuple.
```
>>> m = sc.parallelize([(1, 2), (3, 4)]).values()
>>> m.collect()
[2, 4]
```
variance()										Compute the variance of this RDD’s elements.
```
>>> sc.parallelize([1, 2, 3]).variance()
0.666...
```
zip(other)										Assumes that the two RDDs have the same number of partitions and the same number of elements in each partition (e.g. one was made through a map on the other).
```
>>> x = sc.parallelize(range(0,5))
>>> y = sc.parallelize(range(1000, 1005))
>>> x.zip(y).collect()
[(0, 1000), (1, 1001), (2, 1002), (3, 1003), (4, 1004)]
```
zipWithIndex()									Zips this RDD with its element indices.	The ordering is first based on the partition index and then the ordering of items within each partition. So the first item in the first partition gets index 0, and the last item in the last partition receives the largest index.	This method needs to trigger a spark job when this RDD contains more than one partitions.
```
>>> sc.parallelize(["a", "b", "c", "d"], 3).zipWithIndex().collect()
[('a', 0), ('b', 1), ('c', 2), ('d', 3)]
```
zipWithUniqueId()								Zips this RDD with generated unique Long ids.	Items in the kth partition will get ids k, n+k, 2*n+k, ..., where n is the number of partitions. So there may exist gaps, but this method won’t trigger a spark job, which is different from zipWithIndex
```
>>> sc.parallelize(["a", "b", "c", "d", "e"], 3).zipWithUniqueId().collect()
[('a', 0), ('b', 1), ('c', 4), ('d', 2), ('e', 5)]
```

## 五、 StorageLevel 类
class pyspark.StorageLevel(useDisk, useMemory, useOffHeap, deserialized, replication=1)
Flags for controlling the storage of an RDD. Each StorageLevel records whether to use memory, whether to drop the RDD to disk if it falls out of memory, whether to keep the data in memory in a JAVA-specific serialized format, and whether to replicate the RDD partitions on multiple nodes. Also contains static constants for some commonly used storage levels, MEMORY_ONLY. Since the data is always serialized on the Python side, all the constants use the serialized formats.

DISK_ONLY = StorageLevel(True, False, False, False, 1)
DISK_ONLY_2 = StorageLevel(True, False, False, False, 2)
MEMORY_AND_DISK = StorageLevel(True, True, False, False, 1)
MEMORY_AND_DISK_2 = StorageLevel(True, True, False, False, 2)
MEMORY_AND_DISK_SER = StorageLevel(True, True, False, False, 1)
MEMORY_AND_DISK_SER_2 = StorageLevel(True, True, False, False, 2)
MEMORY_ONLY = StorageLevel(False, True, False, False, 1)
MEMORY_ONLY_2 = StorageLevel(False, True, False, False, 2)
MEMORY_ONLY_SER = StorageLevel(False, True, False, False, 1)
MEMORY_ONLY_SER_2 = StorageLevel(False, True, False, False, 2)
OFF_HEAP = StorageLevel(True, True, True, False, 1)

## 六、 Broadcast 类
class pyspark.Broadcast(sc=None, value=None, pickle_registry=None, path=None)
A broadcast variable created with SparkContext.broadcast(). Access its value through value.

Examples:
```
>>> from pyspark.context import SparkContext
>>> sc = SparkContext('local', 'test')
>>> b = sc.broadcast([1, 2, 3, 4, 5])
>>> b.value
[1, 2, 3, 4, 5]
>>> sc.parallelize([0, 0]).flatMap(lambda x: b.value).collect()
[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
>>> b.unpersist()
>>> large_broadcast = sc.broadcast(range(10000))
```
destroy()									Destroy all data and metadata related to this broadcast variable. Use this with caution; once a broadcast variable has been destroyed, it cannot be used again. This method blocks until destroy has completed.
dump(value, f)
load(path)
unpersist(blocking=False)					Delete cached copies of this broadcast on the executors. If the broadcast is used after this is called, it will need to be re-sent to each executor.
		Parameters:	blocking – Whether to block until unpersisting has completed
value 										Return the broadcasted value

## 七、 Accumulator 类
class pyspark.Accumulator(aid, value, accum_param)
A shared variable that can be accumulated, i.e., has a commutative and associative “add” operation. Worker tasks on a Spark cluster can add values to an Accumulator with the += operator, but only the driver program is allowed to access its value, using value. Updates from the workers get propagated automatically to the driver program.
While SparkContext supports accumulators for primitive data types like int and float, users can also define accumulators for custom types by providing a custom AccumulatorParam object. Refer to the doctest of this module for an example.

add(term)									Adds a term to this accumulator’s value
value										Get the accumulator’s value; only usable in driver program

## 八、 AccumulatorParam 类
class pyspark.AccumulatorParam
Helper object that defines how to accumulate values of a given type.

addInPlace(value1, value2)					Add two values of the accumulator’s data type, returning a new value; for efficiency, can also update value1 in place and return it.
zero(value)									Provide a “zero value” for the type, compatible in dimensions with the provided value (e.g., a zero vector)

## 九、 MarshalSerializer 类
class pyspark.MarshalSerializer
Serializes objects using Python’s Marshal serializer:

http://docs.python.org/2/library/marshal.html
This serializer is faster than PickleSerializer but supports fewer datatypes.

dumps(obj)
loads(obj)

## 十、 PickleSerializer 类
class pyspark.PickleSerializer
Serializes objects using Python’s pickle serializer:

http://docs.python.org/2/library/pickle.html
This serializer supports nearly any Python object, but may not be as fast as more specialized serializers.

dumps(obj)
loads(obj, encoding=None)

## 十一、 PickleSerializer 类
class pyspark.StatusTracker(jtracker)
Low-level status reporting APIs for monitoring job and stage progress.

These APIs intentionally provide very weak consistency semantics; consumers of these APIs should be prepared to handle empty / missing information. For example, a job’s stage ids may be known but the status API may not have any information about the details of those stages, so getStageInfo could potentially return None for a valid stage id.
To limit memory usage, these APIs only provide information on recent jobs / stages. These APIs will provide information for the last spark.ui.retainedStages stages and spark.ui.retainedJobs jobs.

getActiveJobsIds()							Returns an array containing the ids of all active jobs.
getActiveStageIds()							Returns an array containing the ids of all active stages.
getJobIdsForGroup(jobGroup=None)			Return a list of all known jobs in a particular job group. If jobGroup is None, then returns all known jobs that are not associated with a job group.	The returned list may contain running, failed, and completed jobs, and may vary across invocations of this method. This method does not guarantee the order of the elements in its result.
getJobInfo(jobId)							Returns a SparkJobInfo object, or None if the job info could not be found or was garbage collected.
getStageInfo(stageId)						Returns a SparkStageInfo object, or None if the stage info could not be found or was garbage collected.

## 十二、 PickleSerializer 类
class pyspark.SparkJobInfo
Exposes information about Spark Jobs.

## 十三、 PickleSerializer 类
class pyspark.SparkStageInfo
Exposes information about Spark Stages.

## 十四、 PickleSerializer 类
class pyspark.Profiler(ctx)
Note DeveloperApi	PySpark supports custom profilers, this is to allow for different profilers to be used as well as outputting to different formats than what is provided in the BasicProfiler.
A custom profiler has to define or inherit the following methods:
profile - will produce a system profile of some sort. stats - return the collected stats. dump - dumps the profiles to a path add - adds a profile to the existing accumulated profile
The profiler class is chosen when creating a SparkContext
```
>>> from pyspark import SparkConf, SparkContext
>>> from pyspark import BasicProfiler
>>> class MyCustomProfiler(BasicProfiler):
...     def show(self, id):
...         print("My custom profiles for RDD:%s" % id)
...
>>> conf = SparkConf().set("spark.python.profile", "true")
>>> sc = SparkContext('local', 'test', conf=conf, profiler_cls=MyCustomProfiler)
>>> sc.parallelize(range(1000)).map(lambda x: 2 * x).take(10)
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
>>> sc.parallelize(range(1000)).count()
1000
>>> sc.show_profiles()
My custom profiles for RDD:1
My custom profiles for RDD:3
>>> sc.stop()
```
dump(id, path)								Dump the profile into path, id is the RDD id
profile(func)								Do profiling on the function func
show(id)									Print the profile stats to stdout, id is the RDD id
stats()										Return the collected profiling stats (pstats.Stats)

## 十五、 PickleSerializer 类
class pyspark.BasicProfiler(ctx)
BasicProfiler is the default profiler, which is implemented based on cProfile and Accumulator

profile(func)								Runs and profiles the method to_profile passed in. A profile object is returned.
stats()
