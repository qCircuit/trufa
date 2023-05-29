import json
import mlflow
import pickle
import shutil
import torch

import configura
import data
import model_fit
import utilities

if __name__ == '__main__':
    conda_env = utilities.init_enviroment(aws=configura.aws)

    json.dump({
        "use": [k for k,v in configura.features.items()], 
        "object": [k for k,v in configura.features.items() if v=="object"]
    }, open(configura.features_list_path, "w"))

    class NnClickBanner(torch.nn.Module):
        def __init__(self, ninput):
            super(NnClickBanner, self).__init__()
            self.layer_1 = torch.nn.Linear(ninput, 2048) 
            self.layer_2 = torch.nn.Linear(2048, 1024) 
            self.layer_3 = torch.nn.Linear(1024, 512)
            self.layer_4 = torch.nn.Linear(512, 128)
            self.layer_5 = torch.nn.Linear(128, 64)
            self.layer_out = torch.nn.Linear(64, 1) 
            
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(p=0.2)
            self.batchnorm1 = torch.nn.BatchNorm1d(2048)
            self.batchnorm2 = torch.nn.BatchNorm1d(1024)
            self.batchnorm3 = torch.nn.BatchNorm1d(512)
            self.batchnorm4 = torch.nn.BatchNorm1d(128)
            self.batchnorm5 = torch.nn.BatchNorm1d(64)
            
        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.batchnorm1(x)
            x = self.relu(self.layer_2(x))
            x = self.batchnorm2(x)
            x = self.relu(self.layer_3(x))
            x = self.batchnorm3(x)
            x = self.relu(self.layer_4(x))
            x = self.batchnorm4(x)
            x = self.relu(self.layer_5(x))
            x = self.batchnorm5(x)
            x = self.dropout(x)
            x = self.layer_out(x)

            return x
        
    with mlflow.start_run() as run:

        configura.mlflow_pyfunc_model_path += f"_{configura.idx_traffic}_{configura.idx_campaign}"
        shutil.rmtree(configura.mlflow_pyfunc_model_path, ignore_errors=True)

        dftr, dfts, encoder, scaler = data.pipe_processing()
        train_loader = model_fit.get_loader(dftr, True)
        test_loader = model_fit.get_loader(dfts)

        ninput = next(iter(train_loader))[0].shape[1]
        with open(configura.input_size_path, 'wb') as f:
            f.write((ninput).to_bytes(24, byteorder='big', signed=False))
        model = model_fit.torch_fit(NnClickBanner(ninput), train_loader, test_loader)
        probabilities, resulting_metrics = model_fit.evaluate_torch_model(
            model, test_loader
        )

        class TorchBorealis(mlflow.pyfunc.PythonModel):

            def load_context(self, context):
                self.model = NnClickBanner(
                    int.from_bytes(open(context.artifacts['ninput'], 'rb').read(), byteorder='big'))
                self.model.load_state_dict(torch.load(context.artifacts['model'], map_location=torch.device('cpu')))
                self.model.eval()
                self.features = json.load(open(context.artifacts['features'], 'rb'))
                self.encoder = pickle.load(open(context.artifacts['encoder'], 'rb'))
                self.scaler = pickle.load(open(context.artifacts['scaler'], 'rb'))

            def process_device_os(self, model_input):
                model_input.loc[model_input.device_os.str.contains('windo'), 'device_os'] = 'windows'
                model_input.loc[model_input.device_os.str.contains('linux|ubuntu'), 'device_os'] = 'linux'
                model_input.loc[model_input.device_os.str.contains('mac|os x|osx'), 'device_os'] = 'mac'
                model_input.loc[model_input.device_os.str.contains('playstation'), 'device_os'] = 'playstation'
                model_input.loc[model_input.device_os.str.contains('ios'), 'device_os'] = 'ios'
                model_input.loc[model_input.device_os.str.contains('chrome'), 'device_os'] = 'chrome'

                return model_input

            def objects_processing(self, v):
                try:
                    v = str(int(float(v)))
                except ValueError:
                    v = str(v)

                return v.lower()

            def predict(self, context, model_input):
                         
                # assign columns if missing
                model_input[[x for x in self.features['use'] if x not in model_input]] = -1
                model_input = model_input.fillna(-1)
                model_input = model_input[self.features['use']].copy()
                
                # process objects
                for f in self.features["object"]:
                    model_input[f] = model_input[f].apply(self.objects_processing)
                
                # processing
                model_input["site_domain"] = model_input["site_domain"].apply(lambda v: ''.join([x for x in v.split(".") if x != "www"]))
                model_input = self.process_device_os(model_input)
                
                # encode & scale   
                model_input[self.features['object']] = self.encoder.transform(model_input[self.features['object']].values)
                model_input = self.scaler.transform(model_input)
                model_input = torch.Tensor(model_input)
                prediction = torch.sigmoid(self.model(model_input)).detach().numpy()

                return prediction
            
        artifacts = utilities.collect_artifacts()
        artifacts.update({'ninput' : configura.input_size_path})

        mlflow.pyfunc.save_model(
            path = configura.mlflow_pyfunc_model_path,
            python_model = TorchBorealis(),
            artifacts=artifacts,
            conda_env=conda_env
        )
        mlflow.pyfunc.log_model(
            artifact_path = configura.mlflow_pyfunc_model_path,
            python_model = TorchBorealis(),
            artifacts=artifacts,
            conda_env=conda_env
        )

        parameters_to_log = {
            "target": configura.target,
            "aws": configura.aws,
            "batch_size": configura.batch_size,
            "epochs": configura.epochs,
            "learning_rate": configura.learning_rate,
            "test_params": configura.test_params
        }

        mlflow.log_params(parameters_to_log)