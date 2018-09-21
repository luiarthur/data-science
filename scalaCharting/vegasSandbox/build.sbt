name := "vegasSandbox"

version := "0.1.0"

//scalaVersion := "2.11.8"
scalaVersion := "2.12.6"

// FIXME: Not working
libraryDependencies ++= Seq(
  "org.vegas-viz" %% "vegas" % "0.3.12",
  "org.scalatest" %% "scalatest" % "3.0.0" % "test"
)

