import math

def file_stats(pathname):
    avg = stdev = 0
    file = open(pathname, "r")
    nums = []

    for i in file.read().split('\n'):
        nums.append(float(i))

    # calc avg
    for i in range(len(nums)):
        avg += float(nums[i])
    avg /= len(nums)

    # calc stdev
    for i in range(len(nums)):
        stdev += (nums[i]-avg)**2
    stdev = math.sqrt(stdev / (len(nums) - 1))

    return (avg,stdev)


avg, stdev = file_stats("nums.txt")

print('avg: {}\nstdev: {}'.format(avg, stdev))
