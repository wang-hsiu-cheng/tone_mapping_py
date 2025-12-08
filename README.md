# Tone Mapping Software Simulation

## Prepare
- Build Environment
   1. `cd Docker`
   2. `docker compose up`
- Enter Environment
   1. `docker start IClab`
   2. `docker exec -it IClab /bin/bash`
- Compile C++
   - `g++ -I/usr/include/eigen3 -std=c++17 bilateral_filter.cpp -o bilateral_filter`

## Use
- Run Python
- `python3 ./tone_mapping_with_cpp.py`
- After Luminance.txt generated
- `./bilateral_filter`
- Get LDR file
| Remember to delete the .txt files before execute the next hdr-ldr convertion
