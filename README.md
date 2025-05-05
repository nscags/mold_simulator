Mold Simulator

```
Tutorial

# [optional] create virtual environment
python3 -m venv venv
source venv/bin/activate

# clone the repo
git clone https://github.com/nscags/mold_simulator.git

# install requirements
cd mold_simulator/
pip install -r requirements.txt

# run simulator
python3 simulate.py

# Note:
# if you don't have ffpmeg installed
# you can either install it
sudo apt install ffmpeg
# or change the writer in mold_simulator/simulator.py
ani.save(filename, writer='pillow')
# HOWEVER, doing so is much slower
Simulating 100 particles for 1000 frames (w pillow) -> ~6.2 mins
Simulating 100 particles for 1000 frames (w ffmeg)  -> ~2.5 mins
```