#!/usr/bin/env bash

num_executor=100
queue="root.production.offline"
cluster="hadoop-spark2.1"

/home/tools/bin/spark-submit  \
    --cluster $cluster \
    --conf "spark.executor.extraJavaOptions=-XX:+PrintFlagsFinal -XX:+UseParallelOldGC -Xmn2200m -XX:SurvivorRatio=20 -XX:TargetSurvivorRatio=100 -XX:MaxTenuringThreshold=5 -XX:PermSize=96m -XX:MaxPermSize=96m -XX:ReservedCodeCacheSize=128m -XX:-UseBiasedLocking -XX:+ExplicitGCInvokesConcurrent -XX:+PrintTenuringDistribution -XX:PrintFLSStatistics=2 -XX:+PrintGCDetails -XX:+PrintSafepointStatistics -XX:+PrintGCDateStamps -XX:+PrintGCTimeStamps -XX:+PrintGCApplicationStoppedTime -XX:+PrintGCApplicationConcurrentTime -XX:+PrintPromotionFailure -XX:+HeapDumpOnOutOfMemoryError -XX:+UnlockDiagnosticVMOptions" \
    --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
    --conf "spark.kryoserializer.buffer=512m" \
    --conf "spark.blacklist.enabled=true" \
    --conf "spark.shuffle.compress=true" \
    --conf "spark.shuffle.spill.compress=true" \
    --conf "spark.shuffle.io.preferDirectBufs=true" \
    --conf "spark.shuffle.service.enabled=true" \
    --conf "spark.akka.frameSize=1024" \
    --conf "spark.memory.useLegacyMode=true" \
    --conf "spark.shuffle.memoryFraction=0.8" \
    --conf "spark.storage.memoryFraction=0.2" \
    --conf "spark.yarn.executor.memoryOverhead=2048" \
    --conf "spark.shuffle.spill.initialMemoryThreshold=5242880000" \
    --conf "spark.shuffle.memory.estimate.debug.enable=false" \
    --conf "spark.shuffle.spill.checkJvmHeap.enable=true" \
    --conf "spark.shuffle.spill.checkJvmHeap.oldSpacePercent=90" \
    --conf "spark.shuffle.spill.checkJvmHeap.logPercent=88" \
    --conf "spark.rdd.compress=true" \
    --conf "spark.broadcast.compress=true" \
    --conf "spark.driver.maxResultSize=4g" \
    --conf "spark.eventLog.enabled=true" \
    --class org.apache.spark.ml.recommendation.DSGD \
    --master yarn-cluster \
    --queue "$queue" \
    --num-executors "$num_executor" \
    --driver-memory 10000m \
    --executor-memory 13000m \
    ../target/com.example-0.1-SNAPSHOT-with-dependencies.jar  $num_executor 60 100 /user/h_user_profile/liupeng11/ratings.csv 0.1 0.01
