from domain.models.network import Networks, Optimizers, KerasBackbone
from infra.services.blob_storage.google_drive_blob_storage_service import GoogleDriveBlobStorageService
from infra.services.test_case_execution_service_mongodb import TestCaseExecutionServiceMongoDB
from infra.services.test_case_service_mongodb import TestCaseServiceMongoDB


def main():
    serv = TestCaseServiceMongoDB('unet_depth')
    serv_hist = TestCaseExecutionServiceMongoDB('unet_depth')
    networks = [network for network in Networks]
    optimizers = [opt for opt in Optimizers]
    backbones = [bb for bb in KerasBackbone]
    # serv.remove_all()
    # serv.populate(networks, backbones, optimizers)
    # a: TestCase = serv.get('631fc4bf48a461d7f427224e')
    drive = GoogleDriveBlobStorageService()
    _id = drive.save('credentials_old.json')
    drive.download(_id)

    return 0


main()
