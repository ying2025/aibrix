import pandas as pd
import csv

#############################
########### maas  ###########
#############################

# Read the CSV data into a DataFrame
# data = """
# Time,Total,Success,5xx Error,4xx Error
# 2024-10-14 14:04:00,31.6,31.2,,0.387
# 2024-10-14 14:06:00,32.1,31.6,,0.472
# """

# "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/maas/scenario-extraction/2024-10-14 14:04:00/API/qps.csv"

# maas
# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/maas/scenario-extraction/2024-10-14 14:04:00/API/qps.csv" # 45.8 / 16 = 2.8625
# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/maas/scenario-extraction/2024-10-09 01:34:00/API/qps.csv" # 59.0 / 16 = 3.6875
# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/maas/scenario-extraction/2024-10-09 05:34:00/API/qps.csv" # 76.3 / 16 = 4.76875
# Create the DataFrame


# df = pd.read_csv(filename)

# # Find the peak value of the "Total" column
# peak_value = df["Total"].max()

# print(f"The peak value of the 'Total' column is: {peak_value}")


#############################
####### cloudide   ##########
#############################



# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/cloudide/data/cr/request-cr.csv" # 5.433 / 16 = 0.3395625
# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/cloudide/data/bitsai-qa/request-bitsai-qa.csv" # 0.8332 / 16 = 0.052075
# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/cloudide/data/ci-rootcause/request-ci-rootcause.csv" # 0.4329 / 16 = 0.02705625
# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/cloudide/data/ci-solution/request-ci-solution.csv" # 0.1666 / 16 = 0.0104125
# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/cloudide/data/qa-flywheel/request-qa-flywheel.csv" # 8.2984 / 16 = 0.51865
# df = pd.read_csv(filename)
# print(df.columns)
# peak_value = df["bitsai-code-llm-v2.byted.org"].max()
# print(f"The peak value of the 'Total' column is: {peak_value}")


filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/cloudide/data/cr/io-cr.csv" # output 9380.0996 input 325872.3329
# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/cloudide/data/bitsai-qa/io-bitsai-qa.csv" # output 63081.4997 input 11084.833
# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/cloudide/data/ci-rootcause/io-ci-rootcause.csv" # output 145213.03309999997 input 38121.4332
# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/cloudide/data/ci-solution/io-ci-solution.csv" # input 5614.2332
# filename = "/Users/bytedance/Projects/serverless/aibrix-experiment/aibrix-data/internal-trace/cloudide/data/qa-flywheel/io-qa-flywheel.csv" # output 6412.266 input 112725.4653

df = pd.read_csv(filename, parse_dates=['Time'])
df = df.replace("undefined", 0)
df['Time'] = pd.to_datetime(df['Time'], unit = 'ms')  # Ensure timestamp is a datetime object
df = df.set_index('Time')  # Set 'Time' as index for rolling window calculation
sent_columns = df.filter(regex = r'^sent_bytes.rate@')
sent_columns = sent_columns.apply(pd.to_numeric, errors='coerce').fillna(0)
df['sent'] = sent_columns.sum(axis = 1)

recv_columns = df.filter(regex = r'^recv_bytes.rate@')
recv_columns = recv_columns.apply(pd.to_numeric, errors='coerce').fillna(0)
df['recv'] = recv_columns.sum(axis = 1)

peak_value_sent = df['sent'].max()
peak_value_recv = df['recv'] .max()
print(f"peak_value_sent: {peak_value_sent} peak_value_recv {peak_value_recv}")