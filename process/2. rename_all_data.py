# import pandas as pd
#
#
# def rename_columns(df):
#     """Get es results and rename es columns."""
#     df.columns = df.columns.str.replace('_', '')
#
#     for col in df.columns:
#         new_col = col.split('.')
#         if len(new_col) != 1:
#             new_col = new_col[1]
#             print(new_col)
#
#         else:
#             new_col = new_col[0]
#         df = df.rename(columns={col: new_col})
#
#     return df
#
#
# df = pd.read_csv('final_pagination_110422.csv')
# df = rename_columns(df)
# df.to_csv('good_pagination.csv')
# # print(len(df))
