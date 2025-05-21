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

  @ViewChild('cameraImg', { static: false }) cameraImgRef!: ElementRef<HTMLImageElement>;
  imgWidth: number = 300;
  imgHeight: number = 280;

  points: { x: number, y: number }[] = [
    { x: 100, y: 100 },
    { x: 300, y: 100 },
    { x: 300, y: 300 },
    { x: 100, y: 300 }
  ];
  selectedPoint: number | null = null;
  warpedImgUrl: string | null = null;
  draggingPoint: number | null = null;

  constructor(
    private motorsService: MotorsControlService,
    private toastController: ToastController,
    private alertController: AlertController,
    private configService: SetupAPIService
  ) {
    this.streamUrl = this.configService.getNormalStreamUrl();
  }

  ngOnInit() {
    this.fetchKeypoints();
    this.loadHomography();
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
  
  getPolygonPoints(): string {
    // Assumes this.points is an array of objects with x and y properties
    if (!this.points || this.points.length !== 4) {
      return '';
    }
    return this.points.map(p => `${p.x},${p.y}`).join(' ');
  }

  loadHomography() {
    this.configService.getHomography().subscribe({
      next: (res) => {
        if (res.homography) this.homography = res.homography;
      }
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

  onCameraLoad(img: HTMLImageElement) {
    // Aggiorna solo se disponibili valori validi
    if (img.naturalWidth && img.naturalHeight) {
      this.imgWidth = img.naturalWidth;
      this.imgHeight = img.naturalHeight;
    }
  }

  selectPoint(i: number) {
    this.selectedPoint = i;
  }

  // (opzionale) Gestione drag dei punti sull'immagine
  // Puoi aggiungere eventi mouse/touch per spostare i punti direttamente sull'immagine

  calculateHomography() {
    // Chiamata al backend per calcolare la warp e restituire l'immagine post-processata
    this.configService.calculateHomographyFromPoints(this.points).subscribe({
      next: (res) => {
        this.warpedImgUrl = res.warped_url;
      },
      error: () => {
        this.presentToast('Errore nel calcolo omografia', 'danger');
      }
    });
  }

  onPointPointerDown(event: PointerEvent, i: number) {
    event.preventDefault();
    event.stopPropagation();
    this.draggingPoint = i;
    this.selectedPoint = i;
    const svg = (event.target as SVGElement).ownerSVGElement || event.target as SVGSVGElement;
    if (svg && event.pointerId) {
      svg.setPointerCapture(event.pointerId);
    }
  }
  

  onSvgPointerMove(event: PointerEvent) {
    if (this.draggingPoint !== null) {
      event.preventDefault();
      event.stopPropagation();
      const svg = (event.target as SVGElement).ownerSVGElement || event.target as SVGSVGElement;
      const rect = svg.getBoundingClientRect();
      // Calcola coordinate relative allo SVG
      const x = Math.max(0, Math.min(this.imgWidth, event.clientX - rect.left));
      const y = Math.max(0, Math.min(this.imgHeight, event.clientY - rect.top));
      this.points[this.draggingPoint] = { x, y };
    }
  }

  onSvgPointerUp(event?: PointerEvent) {
    if (event && event.pointerId) {
      const svg = (event.target as SVGElement).ownerSVGElement || event.target as SVGSVGElement;
      if (svg) {
        svg.releasePointerCapture(event.pointerId);
      }
    }
    this.draggingPoint = null;
  }
}
