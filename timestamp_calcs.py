import numpy as np
import matplotlib as plt

def calculate_time_differences(timestamps):
    """
    Calculate time differences between consecutive timestamps.

    Parameters:
    timestamps (list): List of timestamps.

    Returns:
    list: List of time differences between consecutive timestamps.
    """
    if len(timestamps) < 2:
        return []
    
    time_differences = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    return time_differences


# timestamps with long response periods removed and non trial period responses left in
timestamps = [0, 13146, 15569, 18194, 18720, 21455, 21991, 24784, 25438, 
              27956, 28746, 31145, 31957, 33639, 34049, 37277, 38067, 
              40288, 40878, 44117, 44452, 46688, 46997, 49050, 49375, 
              51076, 51738, 53301, 53645, 56163, 56416, 59405, 60214, 
              63455, 64201, 66834, 69328, 72717, 74145, 78014, 79477, 
              80405, 82955, 83823, 84869, 86783, 87931, 88390, 91525, 
              91760, 95086, 96227, 99359, 99651, 102772, 103909, 106184, 
              106814, 109239, 109425, 110005, 111685, 111912, 114611, 
              115660, 118441, 118475, 122981, 123363, 125035, 126463, 
              128967, 132195, 134699, 137859, 144537, 144995, 148171, 
              149215, 151616, 152369, 153808, 156285, 157227, 160349, 
              161088, 162746, 163152, 165174, 165847, 167852, 168201, 
              169994, 170314, 173607, 174223, 176532, 177360, 180735, 
              181081, 182988, 183295, 185354, 186150, 188076, 188744, 
              191113, 193617, 196869, 202124, 202499, 205760, 206090, 
              207668, 207956, 211343, 211600, 214218, 214769, 217331, 
              217856, 219557, 220346, 223642, 223985, 226545, 227347, 
              229466, 230205, 233711, 234056, 237012, 237073, 240139, 
              242640, 245890, 247268, 251742, 252123, 253180, 254136, 
              254852, 257780, 258814, 260424, 260465, 261479, 263313, 
              263747, 266339, 266636, 269752, 270660, 271456, 273907, 
              274146, 276216, 276672, 279055, 279570, 281751, 282362, 
              284651, 285396, 289103, 291602, 294651, 300750, 301606, 
              303385, 303717, 305890, 306664, 308957, 309459, 311250, 
              311434, 313988, 314298,316042, 316509, 319696, 320199, 
              323227, 323779, 326030, 326341, 328762, 329757, 331563, 
              331809, 335040, 335346,337760, 340263, 343389, 345889, 
              349484, 353288, 355806, 356054, 358673, 359401, 362703, 
              362773, 364181, 365792, 366538, 368546, 368945, 371806, 
              372178, 374560, 375306, 375615, 378165, 378690, 382086, 
              82352, 383906,384259, 386516, 387458, 390838, 391652, 
              394648, 395492, 398692, 398992, 400437, 401453, 401976, 
              407008, 409511,413070, 415768, 417847, 418225, 421406, 
              422202, 424928, 425182, 427541, 428115, 429084, 430842, 
              431743, 433495, 433806, 436684, 436983, 439429, 439878, 
              442036, 442851, 445459, 445751, 447736, 448568, 450474, 
              450785,453479, 454390, 455404, 457217, 457515, 459527, 
              459804, 461458, 463958]

#timestamps with non trial period responses removed - NOT TRIMMED YET
timestamps_trimmed = [0, 13146, 15569, 18194, 18720, 21455, 21991, 24784, 25438, 
              27956, 28746, 31145, 31957, 33639, 34049, 37277, 38067, 
              40288, 40878, 44117, 44452, 46688, 46997, 49050, 49375, 
              51076, 51738, 53301, 53645, 56163, 56416, 59405, 60214, 
              63455, 64201, 66834, 69328, 72717, 74145, 78014, 79477, 
              80405, 82955, 83823, 84869, 86783, 87931, 88390, 91525, 
              91760, 95086, 96227, 99359, 99651, 102772, 103909, 106184, 
              106814, 109239, 109425, 110005, 111685, 111912, 114611, 
              115660, 118441, 118475, 122981, 123363, 125035, 126463, 
              128967, 132195, 134699, 137859, 144537, 144995, 148171, 
              149215, 151616, 152369, 153808, 156285, 157227, 160349, 
              161088, 162746, 163152, 165174, 165847, 167852, 168201, 
              169994, 170314, 173607, 174223, 176532, 177360, 180735, 
              181081, 182988, 183295, 185354, 186150, 188076, 188744, 
              191113, 193617, 196869, 202124, 202499, 205760, 206090, 
              207668, 207956, 211343, 211600, 214218, 214769, 217331, 
              217856, 219557, 220346, 223642, 223985, 226545, 227347, 
              229466, 230205, 233711, 234056, 237012, 237073, 240139, 
              242640, 245890, 247268, 251742, 252123, 253180, 254136, 
              254852, 257780, 258814, 260424, 260465, 261479, 263313, 
              263747, 266339, 266636, 269752, 270660, 271456, 273907, 
              274146, 276216, 276672, 279055, 279570, 281751, 282362, 
              284651, 285396, 289103, 291602, 294651, 300750, 301606, 
              303385, 303717, 305890, 306664, 308957, 309459, 311250, 
              311434, 313988, 314298,316042, 316509, 319696, 320199, 
              323227, 323779, 326030, 326341, 328762, 329757, 331563, 
              331809, 335040, 335346,337760, 340263, 343389, 345889, 
              349484, 353288, 355806, 356054, 358673, 359401, 362703, 
              362773, 364181, 365792, 366538, 368546, 368945, 371806, 
              372178, 374560, 375306, 375615, 378165, 378690, 382086, 
              82352, 383906,384259, 386516, 387458, 390838, 391652, 
              394648, 395492, 398692, 398992, 400437, 401453, 401976, 
              407008, 409511,413070, 415768, 417847, 418225, 421406, 
              422202, 424928, 425182, 427541, 428115, 429084, 430842, 
              431743, 433495, 433806, 436684, 436983, 439429, 439878, 
              442036, 442851, 445459, 445751, 447736, 448568, 450474, 
              450785,453479, 454390, 455404, 457217, 457515, 459527, 
              459804, 461458, 463958]

time_differences = calculate_time_differences(timestamps)

#print("Timestamps:", timestamps)
#print("Time differences:", time_differences)

average_diff = np.mean(time_differences)
median_diff = np.median(time_differences)

#print(f"\n average diff: {average_diff}, median diff: {median_diff}\n")

# Pair the differences
paired_differences = [(time_differences[i], time_differences[i+1]) for i in range(0, len(time_differences) - 1, 2)]

# Calculate average for each pair
averaged_pairs = [((pair[0] + pair[1]) / 2) for pair in paired_differences]

# Display the averaged pairs
#print(averaged_pairs)

# Define the timeline range based on the minimum and maximum timestamps
min_timestamp = min([min(pair) for pair in paired_differences])
max_timestamp = max([max(pair) for pair in paired_differences])

# Normalize timestamps to fit within [0, 1] for visualization
normalized_pairs = [
    ((start - min_timestamp) / (max_timestamp - min_timestamp), (end - min_timestamp) / (max_timestamp - min_timestamp))
    for start, end in paired_differences
]

# Prepare to plot
plt.figure(figsize=(12, 6))

# Plot each pair on the normalized timeline
for i, (norm_start, norm_end) in enumerate(normalized_pairs):
    plt.plot([norm_start, norm_end], [1, 1], marker='o', markersize=5, label='Event' if i == 0 else "")
    
# Add labels and title
plt.title("Timestamp Events on a Continuous Timeline")
plt.xlabel("Normalized Timestamp")
plt.ylabel("Events")
plt.yticks([])
plt.grid(True)
plt.legend()
plt.show()
