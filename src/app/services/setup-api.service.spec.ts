import { TestBed } from '@angular/core/testing';

import { SetupAPIService } from './setup-api.service';

describe('SetupAPIService', () => {
  let service: SetupAPIService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SetupAPIService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
