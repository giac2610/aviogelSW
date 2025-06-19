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
    origin_x: number,
    origin_y: number,
    hole_spacing_x_mm: number,
    hole_spacing_y_mm: number,
    grid_rows: number,
    grid_cols: number,
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
    maxInertia: number,
    picamera_config: {
      controls: {
        FrameRate: number,
      }
      lores: {
        size: [number, number],
      }
      main: {
        size: [number, number],
      }
    }
    calibration: {
      camera_matrix: number[][],
      distortion_coefficients: number[][],
    },
    calibration_settings: {
      chessboard_cols: number,
      chessboard_rows: number,
      square_size_mm: number,
    },
    fixed_perspective: {
      homography_matrix: number[][],
      output_width: number,
      output_height: number,
    }
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

  executeRoute(route: any[]): Observable<any> {
    const url = `${this.apiUrl}motors/execute-route/`;
    return this.http.post(url, { route });
  }

  initializeCamera(): Observable<any> {
    const url = `${this.apiUrl}camera/initialize-camera/`;
    return this.http.post(url, {});
  }

  deInitializeCamera(): Observable<any> {
    const url = `${this.apiUrl}camera/deinitialize_camera_endpoint/`;
    return this.http.post(url, {});
  }

  setCameraOrigin(origin_x: number, origin_y: number): Observable<any> {
    const url = `${this.apiUrl}camera/set-origin/`;
    return this.http.post(url, { origin_x, origin_y });
  }

  getKeypoints(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}camera/keypoints/`);
  }

  getHomography(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}camera/homography/`);
  }

  getThresholdStreamUrl(): string {
    return `${this.apiUrl}camera/stream/?mode=threshold&keyframe=true`;
  }

  getNormalStreamUrl(): string {
    return `${this.apiUrl}camera/stream/?mode=normal&keyframe=true`;
  }

  getDynamicWarpedStreamUrl(): string {
    // return `${this.apiUrl}camera/dynamic-warped-stream/`;
      return `${this.apiUrl}camera/fixed-perspective-stream/`;
  }

  calibrateCamera(): Observable<any> {
    const url = `${this.apiUrl}camera/calibrate_camera/`;
    return this.http.post(url, {});
  }

  resetCameraCalibration(): Observable<any> {
    const url = `${this.apiUrl}camera/reset-camera-calibration/`;
    return this.http.post(url, {});
  }
  saveFrameCalibration(): Observable<any> {
  const url = `${this.apiUrl}camera/save-frame-calibration/`;
  return this.http.post(url, {});
  }

  getKeypointsCoordinates(): Observable<any> {
    const url = `${this.apiUrl}camera/get_coordinates/`;
    return this.http.get(url);
  }

  getFixedPerspectiveStreamUrl(): string {
    return `${this.apiUrl}camera/fixed-perspective-stream/`;
  }

  setFixedPerspectiveView(): Observable<any> {
    const url = `${this.apiUrl}camera/set-fixed-perspective/`;
    return this.http.post(url, {});
  }

  getMotorsRoute(): Observable<any> {
    const url = `${this.apiUrl}camera/get_route/`;
    return this.http.get(url);
  }



}
