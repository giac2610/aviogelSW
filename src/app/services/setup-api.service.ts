import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Settings {
  motors: {
    extruder:{
      homeDir: 0 | 1,
      stepOneRev: number,
      microstep:  2 | 4 | 8 | 16 | 32 | 64 | 128,
      pitch: number,
      maxSpeed: number,
      acceleration: number,
      deceleration: number,
      maxTravel: number,
      hertz: number
    },
    conveyor:{
      homeDir: 0 | 1,
      stepOneRev: number,
      microstep:  2 | 4 | 8 | 16 | 32 | 64 | 128,
      pitch: number,
      maxSpeed: number,
      acceleration: number,
      deceleration: number,
      maxTravel: number,
      hertz: number
    },
    syringe:{
      homeDir: 0 | 1,
      stepOneRev: number,
      microstep:  2 | 4 | 8 | 16 | 32 | 64 | 128,
      pitch: number,
      maxSpeed: number,
      acceleration: number,
      deceleration: number,
      maxTravel: number,
      hertz: number
    }

  }

  camera: {
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
  
  // Ottieni le impostazioni macchina
  getSettings(): Observable<Settings>{
    const url = `${this.apiUrl}config/get/`
    return this.http.get<Settings>(url);
    }
  
    updateSettings(newSettings: Settings): Observable<Settings> {
      // console.log("Dati inviati al backend:", newSettings);
      return this.http.post<Settings>(`${this.apiUrl}config/update/`, newSettings);
    }
}
