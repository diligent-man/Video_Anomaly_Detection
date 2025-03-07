There are 2 configuration versions for training on two corresponding operating system (os), namely
ubuntu 24.04 (debian-base linux) and Win 11. Both of them are run on the same machine with the following resources:
    1/ Storage: 4TB SSD
    2/ RAM: 64GB, bus 4800MT/s
    3/ CPU: i9-13900HX
    4/ GPU: RTX 4090 mobile

Some minor tweaks were applied for utilizing multiprocessing mechanism in pytorch dataloader.
        OS      |       Tweaks
----------------|--------------------------------
     Ubuntu     | Swap memory is set to 24GB
     Win 11     | Virtual memory is set to 80GB
