import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class MotorsControlService {
  private apiUrl = `http://${window.location.hostname}:8000/motors/` 
  
  constructor(private http: HttpClient) { }
  
  // Ottieni le impostazioni macchina
  updateSettings():Observable<any>{
    const url = `${this.apiUrl}update/`
    return this.http.get<any>(url);
    }
  
    // Aggiorna le impostazioni macchina
    moveMotor(body: any):Observable<any> {
      // console.log(body)
      const url = `${this.apiUrl}move/`
      return this.http.post<any>(url, body);
    }

    stopMotor(body: any):Observable<any>{
      return this.http.post<any>(`${this.apiUrl}stop/`, body);
    }
}
