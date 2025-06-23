import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { MotorsControlService } from '../../services/motors-control.service';
import { ToastController, AlertController } from '@ionic/angular';
import { SetupAPIService } from 'src/app/services/setup-api.service';
import { switchMap, tap, catchError } from 'rxjs/operators';
import { of } from 'rxjs';

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
  // graphUrl: string = 'http://localhost:8000/camera/plot_graph/';
  graphUrl: string = '';

  @ViewChild('cameraImg', { static: false }) cameraImgRef!: ElementRef<HTMLImageElement>;
  imgWidth: number = 1280/3.5;
  imgHeight: number = 720/3.5;

  constructor(
    private toastController: ToastController,
    private configService: SetupAPIService,
    private motorsControlService: MotorsControlService,
  ) {
    this.streamUrl = this.configService.getNormalStreamUrl();
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
    this.getRealCoordinates();
  }


  async presentToast(message: string, color: string = 'success') {
    const toast = await this.toastController.create({
      message: message,
      duration: 1400,
      color: color
    });
    await toast.present();
  }

  getRealCoordinates(){
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

  delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
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
    this.configService.getMotorsRoute().subscribe({
      next: (res) => {
        if (res.status === 'success') {
          this.presentToast('Rotta ottenuta con successo');
          
          // ORA QUESTO FUNZIONA PERFETTAMENTE!
          // Passiamo direttamente l'array, perché il servizio sa come gestirlo.
          this.configService.executeRoute(res.motor_commands).subscribe({
              next: (moveRes) => {
                if (moveRes.status === 'success') {
                  console.log(`Comandi eseguiti con successo`);
                  this.graphUrl = 'http://localhost:8000/camera/plot_graph/?t=' + new Date().getTime();
                } else {
                  this.presentToast(`Errore nell'esecuzione: ${moveRes.message}`, 'danger');
                }
              }, 
              error: (err) => {
                this.presentToast(`Errore nella richiesta di esecuzione: ${err.message}`, 'danger');
              }
            });

          if (res.plot_graph_base64) {
            this.graphUrl = 'data:image/png;base64,' + res.plot_graph_base64;
          }
        } else {
          this.presentToast(res.message || 'Errore nell\'ottenimento della rotta', 'danger');
        }
      },
      error: (_err: any) => {
        this.presentToast('Errore nella richiesta della rotta', 'danger');
      }
    });
  }
}
