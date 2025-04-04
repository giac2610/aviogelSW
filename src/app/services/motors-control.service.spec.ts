import { MotorsControlService } from './motors-control.service';
import { TestBed } from '@angular/core/testing';

describe('MotorsControlService', () => {
  let service: MotorsControlService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(MotorsControlService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
