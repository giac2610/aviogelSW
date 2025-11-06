import { LedService } from './../../services/led.service';
import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { ToastController } from '@ionic/angular';
import { SetupAPIService } from 'src/app/services/setup-api.service';
import { switchMap } from 'rxjs/operators';

@Component({
  selector: 'app-blob-simulation',
  templateUrl: './blob-simulation.page.html',
  styleUrls: ['./blob-simulation.page.scss'],
  standalone: false,
})
export class BlobSimulationPage implements OnInit {
  keypoints: [number, number][] = [];
  coordinates: [number, number][] = [];
  error: string | null = null;
  streamUrl: string = '';
  graphUrl: string = '';
  isCameraInit: boolean = false;

  @ViewChild('cameraImg', { static: false }) cameraImgRef!: ElementRef<HTMLImageElement>;
  imgWidth: number = 1280 / 3.5;
  imgHeight: number = 720 / 3.5;

  constructor(
    private toastController: ToastController,
    private configService: SetupAPIService,
    private ledService: LedService
  ) {
    this.streamUrl = this.configService.getNormalStreamUrl();
  }

  ngOnInit() {
  }


  async presentToast(message: string, color: string = 'success') {
    const toast = await this.toastController.create({
      message: message,
      duration: 1400,
      color: color
    });
    await toast.present();
  }

  getRealCoordinates() {
    this.configService.getKeypointsCoordinates().subscribe({
      next: (res) => {
        this.presentToast('Coordinate reali recuperate con successo');
        console.log('Coordinate reali:', res.coordinates);
        this.coordinates = res.coordinates;
        this.graphUrl = 'http://localhost:8000/camera/plot_graph/?t=' + new Date().getTime();
      },
      error: () => {
        this.presentToast('Errore nel recupero delle coordinate reali', 'danger');
      }
    });
  }

  viewRoute() {
    this.configService.getMotorsRoute().subscribe({
      next: (res) => {
        if (res.status === 'success') {
          this.presentToast('Rotta ottenuta con successo');
          console.log('percorso:', res.motor_commands);
          // Se la risposta contiene l'immagine base64:
          if (res.plot_graph_base64) {
            this.graphUrl = 'data:image/png;base64,' + res.plot_graph_base64;
          }
        } else {
          this.presentToast(res.message || 'Errore nell\'esecuzione della rotta', 'danger');
        }
      },
      error: (_err: any) => {
        this.presentToast('Errore nella richiesta della rotta', 'danger');
      }
    });
  }

  executeRoute() {
    this.configService.initializeCamera().pipe(
      switchMap(() => this.configService.getMotorsRoute()),
      switchMap((res) => {
        if (res.status === 'success') {
          console.log('route to execute:', res.motor_commands);
          this.ledService.startGreenLoading().subscribe({
            next: () => this.presentToast('Green loading avviato', 'success'),
            error: () => this.presentToast('Errore nell\'avviare la wave', 'danger')
          });;
          return this.configService.executeRoute(res.motor_commands);

        } else {
          throw new Error(res.message || 'Errore nell\'ottenimento della rotta');
        }
      }),
      switchMap(() => this.configService.deInitializeCamera())
    ).subscribe({
      next: () => {
        // this.ledSerivce.startWaveEffect().subscribe({
        //   next: () => this.presentToast('wave avviata', 'success'),
        //   error: () => this.presentToast('Errore nell\'avviare la wave', 'danger')
        // });;; this.presentToast('Rotta eseguita con successo');
      },
      error: (err) => { this.presentToast(err.message, 'danger'); }
    });
  }

  stopMotors() {
    this.configService.stopMotors().subscribe({
        next: (response) => {
            this.presentToast(`Successo: ${response.status}`, 'success');
        },
        error: (error) => {
            this.presentToast(`Errore: ${error.error.detail || error.message}`, 'danger');
        }
    });
  }

  viewMoldPosition() {
    this.ledService.startMoldPositioning().subscribe({
      next: () => this.presentToast('Mold positioning avviato', 'success'),
      error: () => this.presentToast('Errore nell\'avviare il mold positioning', 'danger')
    });
  }
}
