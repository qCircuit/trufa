features = {
    'device_devicetype': "object",
    'device_make': "object",
    'device_model': "object",
    'device_os': "object",
    'device_osv': "object",
    'imp_tagid': "object",
    'site_domain': "object",
    'site_id': "object",
    'site_publisher_id': "object",
    'site_traffic_type': "object",
    "ssp_id": "object",
    'seatbid_bid_crid': "object",
    'template_id': "object",
    'count_template_slot': "object",
    "hour": "object"
}
target = {"is_click": "bool"}

aws = False
batch_size = 64
epochs = 1
learning_rate = 0.001
test_params = [3000, 5, 0.005]

creds_path = "dicts/creds.json"
data_path = "data/"
encoder_path = "encoder.pkl"
features_list_path = "features_list.json"
idx = "conjuntoDeDatosManual"
idx_campaign = 0
idx_traffic = "banner"
input_size_path = "input_size.bin"
logs_path = "logs.log"
mlflow_pyfunc_model_path = "model_mlflow_pyfunc"
model_path = "model.pth"
probabilities_path = "probabilities.csv"
scaler_path = "scaler.pkl"
test_plot_path = "test_plot.png"

