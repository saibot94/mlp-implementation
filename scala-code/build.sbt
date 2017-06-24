name := "data-mining-mlp"

version := "1.0"

scalaVersion := "2.11.11"

libraryDependencies ++= Seq(
  "org.scalatest" % "scalatest_2.11" % "3.0.1",
  "org.apache.spark" %% "spark-core" % "2.1.1",
  "org.apache.spark" %% "spark-mllib" % "2.1.1",
  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalanlp" %% "breeze-viz" % "0.12"
)
resolvers ++= Seq(
    "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

mainClass in assembly := Some("com.cristis.mlp.TrainMNISTDataset")
