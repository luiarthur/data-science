name := "smileSandbox"

version := "0.1.0"

scalaVersion := "2.12.7"

libraryDependencies ++= Seq(
  "com.github.haifengl" %% "smile-scala" % "1.5.1",
  "org.scalatest" %% "scalatest" % "3.0.0" % "test"
)

/* As per https://haifengl.github.io/smile/faq.html
You need to add:

addSbtPlugin("com.jsuereth" % "sbt-pgp" % "1.0.0")

to "~/.sbt/0.13/plugins/gpg.sbt"

*/
