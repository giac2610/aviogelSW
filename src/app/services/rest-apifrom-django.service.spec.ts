import { TestBed } from '@angular/core/testing';

import { RestAPIfromDjangoService } from './rest-apifrom-django.service';

describe('RestAPIfromDjangoService', () => {
  let service: RestAPIfromDjangoService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(RestAPIfromDjangoService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
