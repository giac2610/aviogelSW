import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Settings {
  motors: {
    extruder:{
      stepOneRev: number,
      microstep:  2 | 4 | 8 | 16 | 32 | 64 | 128,
      maxSpeed: number,
      acceleration: number,
      deceleration: number
    },
    conveyor:{
      stepOneRev: number,
      microstep:  2 | 4 | 8 | 16 | 32 | 64 | 128,
      maxSpeed: number,
      acceleration: number,
      deceleration: number
    },
    syringe:{
      stepOneRev: number,
      microstep:  2 | 4 | 8 | 16 | 32 | 64 | 128,
      maxSpeed: number,
      acceleration: number,
      deceleration: number
    }

  }

  camera: {
    minThreshold: number,
    maxThreshold: number,
    areaFilter: boolean,
    minArea: number,
    maxArea: number,
    circularityFitler: boolean,
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
  getSettings(): Observable<Settings[]>{
    const url = `${this.apiUrl}config/get/`
    console.log("url: ", url)
    return this.http.get<Settings[]>(url);
    }
  
    // Aggiorna le impostazioni macchina
    updateSettings(newSettings: any): Observable<Settings[]> {
      return this.http.post<Settings[]>(`${this.apiUrl}update/`, newSettings);
    }
}
