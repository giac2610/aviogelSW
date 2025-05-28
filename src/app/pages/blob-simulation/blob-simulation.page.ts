import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { MotorsControlService } from '../../services/motors-control.service';
import { ToastController, AlertController } from '@ionic/angular';
import { SetupAPIService } from 'src/app/services/setup-api.service';

@Component({
  selector: 'app-blob-simulation',
  templateUrl: './blob-simulation.page.html',
  styleUrls: ['./blob-simulation.page.scss'],
  standalone: false,
})
export class BlobSimulationPage implements OnInit {
  keypoints: [number, number][] = [];
  coordinates: [number, number][] = [];
  loading = false;
  error: string | null = null;
  streamUrl: string;
  homography: number[][] = [
    [1,0,0],
    [0,1,0],
    [0,0,1]
  ];
  reconstructGridStatus: string | null = null;
  boundingBoxVertices: [number, number][] = [];
  parallelepipedVertices: [number, number][] = [];
  dynamicWarpedStreamUrl: string;

  @ViewChild('cameraImg', { static: false }) cameraImgRef!: ElementRef<HTMLImageElement>;
  imgWidth: number = 300;
  imgHeight: number = 280;

  constructor(
    private toastController: ToastController,
    private alertController: AlertController,
    private configService: SetupAPIService
  ) {
    this.streamUrl = this.configService.getNormalStreamUrl();
    this.dynamicWarpedStreamUrl = this.configService.getDynamicWarpedStreamUrl();
  }

  ngOnInit() {
    this.configService.initializeCamera().subscribe({
      next: (res) => {
        if (res.success) {
          this.presentToast('Camera inizializzata con successo');
        } else {
          this.presentToast('Errore nell\'inizializzazione della camera', 'danger');
        }
      },
    });
    this.fetchKeypoints();
    this.loadHomography();
    this.getRealCoordinates();
  }

  fetchKeypoints() {
    this.loading = true;
    this.error = null;
    this.configService.getKeypoints().subscribe({
      next: (res) => {
        this.keypoints = res.keypoints || [];
        this.boundingBoxVertices = res.bounding_box_vertices || [];
        this.parallelepipedVertices = res.parallelepiped_vertices || [];
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Errore nel recupero dei keypoints';
        this.loading = false;
      }
    });
  }

  loadHomography() {
    this.configService.getSettings().subscribe({
      next: (res) => {
        console.log('Settings:', res);
        if (res.camera.calibration.camera_matrix) {
          this.homography = res.camera.calibration.camera_matrix;
        } else {
          this.presentToast('Nessuna omografia disponibile');
        }
      }
    });
  }

  async presentToast(message: string, color: string = 'success') {
    const toast = await this.toastController.create({
      message: message,
      duration: 1400,
      color: color
    });
    await toast.present();
  }

  onCameraLoad(img: HTMLImageElement) {
    // Aggiorna solo se disponibili valori validi
    if (img.naturalWidth && img.naturalHeight) {
      this.imgWidth = img.naturalWidth;
      this.imgHeight = img.naturalHeight;
    }
  }

  calibrateCamera() {
   this.configService.calibrateCamera().subscribe({
      next: (res) => {
        if (res.status === 'success') {
          this.presentToast(res.message || 'Calibrazione completata con successo', 'success');
          this.fetchKeypoints();
        } else {
          this.presentToast(res.message || 'Errore nella calibrazione', 'danger');
        }
      } 
    })
  }

  saveFrameCalibration() {
    this.configService.saveFrameCalibration().subscribe({
      next: (res) => {
        if (res.status === 'success') {
          this.presentToast('Frame salvato: ' + res.filename);
        } else {
          this.presentToast('Errore nel salvataggio frame', 'danger');
        }
      },
      error: () => {
        this.presentToast('Errore nel salvataggio frame', 'danger');
      }
    });
  }

  getRealCoordinates(){
    this.configService.getKeypointsCoordinates().subscribe({
      next: (res) => {
          this.presentToast('Coordinate reali recuperate con successo');
          console.log('Coordinate reali:', res.coordinates);
          this.coordinates = res.coordinates;
        
      },
      error: () => {
        this.presentToast('Errore nel recupero delle coordinate reali', 'danger');
      }
    });
  }

  setFixedPerspectiveView() {
    this.configService.setFixedPerspectiveView().subscribe({
      next: (res) => {
        if (res.status === 'success') {
          this.presentToast('Vista prospettica fissa impostata con successo');
        } else {
          this.presentToast(res.message || 'Errore nell\'impostazione della vista prospettica', 'danger');
        }
      }
    });
  }
}
