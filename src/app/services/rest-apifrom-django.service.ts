import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface User {
  expertUser: boolean;
  id: number;
  name: string;
  gender: string;
  avatar_url: string;
}

@Injectable({
  providedIn: 'root'
})
export class RestAPIfromDjangoService {
  // private apiUrl = 'http://localhost:8000/'; // URL base delle API Django
  private apiUrl = `http://${window.location.hostname}:8000/` //

  constructor(private http: HttpClient) {}

  // Metodo per ottenere la lista degli utenti
  getUsers(): Observable<User[]> {
    const url = `${this.apiUrl}users/list/`;
    return this.http.get<User[]>(url);
  }

  // Metodo per aggiungere un nuovo utente
  addUser(name: string, gender: string): Observable<User> {
    const url = `${this.apiUrl}users/add/`;
    const body = { name, gender };
    return this.http.post<User>(url, body);
  }
}