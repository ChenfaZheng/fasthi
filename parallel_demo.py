from ParallelCommander import Manager


manager = Manager(config='./config-linux.yml')
fits_list = manager.get_fits_list()

# check fits file in queue
# for idx, fitsname in enumerate(fits_list):
#     print('{:>5d} {:s}'.format(idx, fitsname))


ctr = 0
for idx, fitsname in enumerate(fits_list):
    if idx < 2:
        continue
    ctr += 1
    # only run 2 fits file for test
    if ctr > 2:
        break
    print('------------------------------------------------------')
    print('{:d}/{:d} {:s}'.format(ctr, len(fits_list), fitsname))
    print('------------------------------------------------------')
    manager.run_one_fits(idx)