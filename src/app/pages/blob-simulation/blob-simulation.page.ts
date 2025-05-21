import { Component, OnInit } from '@angular/core';
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
  step: number = 1;
  keypoints: [number, number][] = [];
  loading = false;
  error: string | null = null;
  streamUrl: string;
  selectedKeypoint: [number, number] | null = null;
  conveyorMoved = false;
  extruderMoved = false;
  homography: number[][] = [
    [1,0,0],
    [0,1,0],
    [0,0,1]
  ];
  useWarpedStream: boolean = false;
  staticOriginalUrl: string | null = null;
  staticWarpedUrl: string | null = null;
  staticHomography: number[][] | null = null;
  contourParams = {
    minThreshold: 44,
    maxThreshold: 251,
    minArea: 70,
    maxArea: 1000
    // aggiungi altri parametri se vuoi
  };
  contourParamsSaved = false;
  contourStreamUrl: string;
  contourHomography: number[][] | null = null;
  selectedStream: string = 'normal';

  constructor(
    private motorsService: MotorsControlService,
    private toastController: ToastController,
    private alertController: AlertController,
    private configService: SetupAPIService
  ) {
    this.streamUrl = this.configService.getNormalStreamUrl();
    this.contourStreamUrl = this.getContourStreamUrl();
  }

  ngOnInit() {
    this.fetchKeypoints();
    this.loadHomography();
    this.loadContourParams();
    this.updateContourStreamUrl();
  }

  fetchKeypoints() {
    this.loading = true;
    this.error = null;
    this.configService.getKeypoints().subscribe({
      next: (res) => {
        this.keypoints = res.keypoints || [];
        this.selectedKeypoint = this.getClosestKeypoint();
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Errore nel recupero dei keypoints';
        this.loading = false;
      }
    });
  }

  loadHomography() {
    this.configService.getHomography().subscribe({
      next: (res) => {
        if (res.homography) this.homography = res.homography;
      }
    });
  }

  loadContourParams() {
    this.configService.getContourParams().subscribe(params => {
      this.contourParams = { ...this.contourParams, ...params };
    });
  }

  updateContourParams() {
    this.configService.updateContourParams(this.contourParams).subscribe(() => {
      this.contourParamsSaved = true;
      setTimeout(() => this.contourParamsSaved = false, 1500);
    });
  }

  fetchContourHomography() {
    this.configService.getContourHomography().subscribe(res => {
      this.contourHomography = res.homography;
    });
  }

  getClosestKeypoint(): [number, number] | null {
    if (!this.keypoints.length) return null;
    let minDist = Infinity;
    let closest: [number, number] | null = null;
    for (const kp of this.keypoints) {
      const dist = Math.sqrt(kp[0] * kp[0] + kp[1] * kp[1]);
      if (dist < minDist) {
        minDist = dist;
        closest = kp;
      }
    }
    return closest;
  }

  async confirmStep() {
    if (this.step === 1) {
      this.step = 2;
    } else if (this.step === 2) {
      if (!this.selectedKeypoint) {
        this.presentToast('Nessun keypoint disponibile', 'danger');
        return;
      }
      const conveyorTarget = -Math.abs(this.selectedKeypoint[1]);
      this.motorsService.moveMotor({ targets: { conveyor: conveyorTarget } }).subscribe({
        next: () => {
          this.presentToast('Conveyor mosso', 'success');
          this.conveyorMoved = true;
          this.step = 3;
        },
        error: () => this.presentToast('Errore nel movimento conveyor', 'danger')
      });
    } else if (this.step === 3) {
      this.motorsService.moveMotor({ targets: { extruder: 10 } }).subscribe({
        next: () => {
          this.presentToast('Extruder mosso', 'success');
          this.extruderMoved = true;
          this.step = 4;
        },
        error: () => this.presentToast('Errore nel movimento extruder', 'danger')
      });
    }
  }

  async presentToast(message: string, color: string = 'success') {
    const toast = await this.toastController.create({
      message: message,
      duration: 1400,
      color: color
    });
    await toast.present();
  }

  async resetSimulation() {
    this.step = 1;
    this.conveyorMoved = false;
    this.extruderMoved = false;
    this.fetchKeypoints();
  }

  toggleWarpedStream() {
    this.useWarpedStream = !this.useWarpedStream;
    this.streamUrl = this.useWarpedStream
      ? this.configService.getDynamicWarpedStreamUrl()
      : this.configService.getNormalStreamUrl();
  }

  captureAndWarpFrame() {
    this.configService.captureAndWarpFrame().subscribe({
      next: (res) => {
        this.staticOriginalUrl = res.original_url;
        this.staticWarpedUrl = res.warped_url;
        this.staticHomography = res.homography;
      }
    });
  }

  updateContourStreamUrl() {
    // Aggiungi il parametro mode all'URL dello stream contour
    this.contourStreamUrl = this.getContourStreamUrl();
  }

  getContourStreamUrl(): string {
    return this.selectedStream === 'threshold'
      ? this.configService.getContourStreamUrl() + '?mode=threshold'
      : this.configService.getContourStreamUrl() + '?mode=normal';
  }
}
