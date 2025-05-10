from roboflow import Roboflow
rf = Roboflow(api_key="BCU3fopWID6eXzrQAwZR")
project = rf.workspace("vision-3gxqu").project("sugarbeets-zg7nc")
version = project.version(2)
dataset = version.download("yolov12")
                