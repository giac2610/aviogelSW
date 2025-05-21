import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Settings {
  motors: {
    extruder:{
      homeDir: 0 | 1,
      stepOneRev: number,
      microstep: 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128,
      pitch: number,
      maxSpeed: number,
      acceleration: number,
      deceleration: number,
      maxTravel: number,
      hertz: number,
      stepsPerMm: number
    },
    conveyor:{
      homeDir: 0 | 1,
      stepOneRev: number,
      microstep: 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128,
      pitch: number,
      maxSpeed: number,
      acceleration: number,
      deceleration: number,
      maxTravel: number,
      hertz: number,
      stepsPerMm: number
    },
    syringe:{
      homeDir: 0 | 1,
      stepOneRev: number,
      microstep:  1 | 2 | 4 | 8 | 16 | 32 | 64 | 128,
      pitch: number,
      maxSpeed: number,
      acceleration: number,
      deceleration: number,
      maxTravel: number,
      hertz: number,
      stepsPerMm: number
    }

  }

  camera: {
    x: number,
    y: number,
    minThreshold: number,
    maxThreshold: number,
    areaFilter: boolean,
    minArea: number,
    maxArea: number,
    circularityFilter: boolean,
    minCircularity: number,
    maxCircularity: number,
    inertiaFilter: boolean,
    minInertia: number,
    maxInertia: number
  }
}

@Injectable({
  providedIn: 'root'
})
export class SetupAPIService {
  private apiUrl = `http://${window.location.hostname}:8000/` 
  
  constructor(private http: HttpClient) { }
  
  getSettings(): Observable<Settings>{
    const url = `${this.apiUrl}config/get/`
    return this.http.get<Settings>(url);
  }

  updateSettings(newSettings: Settings): Observable<Settings> {
    return this.http.post<Settings>(`${this.apiUrl}config/update/`, newSettings);
  }

  updateCameraSettings(cameraSettings: Settings['camera']): Observable<any> {
    const url = `${this.apiUrl}camera/update-camera-settings/`;
    return this.http.post(url, cameraSettings);
  }

  stopMotors(): Observable<any> {
    const url = `${this.apiUrl}motors/stop/`;
    return this.http.post(url, {});
  }

  moveMotor(targets: { [key: string]: number }): Observable<any> {
    const url = `${this.apiUrl}motors/move/`;
    return this.http.post(url, { targets: { ...targets } });
  }

  getCurrentSpeeds(): Observable<{ syringe: number; extruder: number; conveyor: number }> {
    const url = `${this.apiUrl}motors/speeds/`;
    return this.http.get<{ syringe: number; extruder: number; conveyor: number }>(url);
  }

  setCameraOrigin(x: number, y: number): Observable<any> {
    const url = `${this.apiUrl}camera/set-origin/`;
    return this.http.post(url, { x, y });
  }

  getKeypoints(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}camera/keypoints/`);
  }

  getKeypointsAll(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}camera/keypoints-all/`);
  }

  getHomography(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}camera/homography/`);
  }

  getFrame(): Observable<Blob> {
    return this.http.get(`${this.apiUrl}camera/frame/`, { responseType: 'blob' });
  }

  getThresholdStreamUrl(): string {
    return `${this.apiUrl}camera/stream/?mode=threshold&keyframe=true`;
  }

  getNormalStreamUrl(): string {
    return `${this.apiUrl}camera/stream/?mode=normal&keyframe=true`;
  }

  getDynamicWarpedStreamUrl(): string {
    return `${this.apiUrl}camera/dynamic-warped-stream/`;
  }

  captureAndWarpFrame(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}camera/capture-and-warp-frame/`);
  }

  calculateHomographyFromPoints(points: {x: number, y: number}[]): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}camera/calculate-homography-from-points/`, { points });
  }
}
